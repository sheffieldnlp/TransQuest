# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for token classification.
"""

import logging
import os

import numpy as np
from datasets import ClassLabel, load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import is_main_process

from transquest.evaluation.wordlevel import compute_scores
from transquest_cli.model_args import ModelArguments
from transquest_cli.data_args import DataTrainingArguments
from transquest_cli.utils import DATASETS_LOADERS_DIR

logger = logging.getLogger(__name__)


def setup_logging(training_args):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)


def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list


def get_true_predictions(raw_predictions, raw_labels, true_length_src, label_list, labels_in_gaps):
    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(raw_predictions, raw_labels)
    ]
    # Split in src and mt
    preds_src, preds_mt = [], []
    for preds, src_length in zip(true_predictions, true_length_src):
        preds_src.append(preds[:src_length])
        # For the target side we predict OK by default for the GAPS
        preds_mt_no_gaps = preds[src_length:]
        if labels_in_gaps:
            # TODO: Remove the direct use of OK label
            preds_mt_with_gaps = ["OK"] * (2 * len(preds_mt_no_gaps) + 1)
            for i, tag in enumerate(preds_mt_no_gaps):
                preds_mt_with_gaps[2 * i + 1] = tag
            preds_mt.append(preds_mt_with_gaps)
        else:
            preds_mt.append(preds_mt_no_gaps)

    return preds_src, preds_mt


# Tokenize all texts and align the labels with them.
def tokenize_and_align_labels(examples, tokenizer, padding, label_to_id, label_all_tokens, labels_in_gaps):
    tokenized_inputs = tokenizer(
        text=examples["src"],
        text_pair=examples["mt"],
        padding=padding,
        truncation=True,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
        return_offsets_mapping=True,
    )
    tokenized_inputs["length_source"] = [len(label_src) for label_src in examples["src_tags"]]
    offset_mappings = tokenized_inputs.pop("offset_mapping")
    input_ids_all = tokenized_inputs["input_ids"]
    labels = []
    for label_src, label_mt, input_ids, offset_mapping in zip(
        examples["src_tags"], examples["mt_tags"], input_ids_all, offset_mappings
    ):
        # remove the labels for GAPS in mt
        if labels_in_gaps:
            label_mt = [l for i, l in enumerate(label_mt) if i % 2 != 0]
        label = label_src + label_mt
        label_index = 0
        current_label = -100
        label_ids = []
        for input_id, offset in zip(input_ids, offset_mapping):
            # We set the label for the first token of each word. Special characters will have an offset of (0, 0)
            # so the test ignores them.
            # TODO: Very specific to XLM-Roberta tokenization. How to generalise?
            if offset[0] == 0 and offset[1] != 0 and tokenizer.convert_ids_to_tokens(input_id) != "▁":
                current_label = label_to_id[label[label_index]]
                label_index += 1
                label_ids.append(current_label)
            # For special tokens, we set the label to -100 so it's automatically ignored in the loss function.
            elif offset[0] == 0 and offset[1] == 0:
                label_ids.append(-100)
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(current_label if label_all_tokens else -100)
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def predict_and_save_output(
    trainer, tokenized_eval_dataset, label_list, labels_in_gaps, output_eval_results_file, output_eval_predictions_file
):
    raw_predictions, raw_labels, _ = trainer.predict(tokenized_eval_dataset)
    raw_predictions = np.argmax(raw_predictions, axis=2)

    preds_src, preds_mt = get_true_predictions(
        raw_predictions,
        raw_labels,
        tokenized_eval_dataset["length_source"],
        label_list=label_list,
        labels_in_gaps=labels_in_gaps,
    )

    f1_bad_src, f1_good_src, mcc_src = compute_scores(
        [[label_list[tag] for tag in tags] for tags in tokenized_eval_dataset["src_tags"]], preds_src
    )

    f1_bad_tgt, f1_good_tgt, mcc_tgt = compute_scores(
        [[label_list[tag] for tag in tags] for tags in tokenized_eval_dataset["mt_tags"]], preds_mt
    )
    metrics = {
        "src_f1-bad": f1_bad_src,
        "src_f1-good": f1_good_src,
        "src_mcc": mcc_src,
        "mt_f1-bad": f1_bad_tgt,
        "mt_f1-good": f1_good_tgt,
        "mt_mcc": mcc_tgt,
    }

    if trainer.is_world_process_zero():
        with open(output_eval_results_file, "w") as writer:
            for key, value in metrics.items():
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        # Save predictions
        with open(output_eval_predictions_file, "w") as writer:
            for prediction in preds_src:
                writer.write(" ".join(prediction) + "\n")


def main(model_args, data_args, training_args):
    # Set seed before initializing model
    set_seed(training_args.seed)

    # Load the dataset
    if data_args.synthetic_train_dir:
        synthetic_train_data = load_dataset(
            f"{DATASETS_LOADERS_DIR}/{data_args.dataset_name}", data_dir=data_args.synthetic_train_dir, split="train"
        )
        features = synthetic_train_data.features

    if training_args.do_train or training_args.do_eval or training_args.do_predict:
        datasets = load_dataset(f"{DATASETS_LOADERS_DIR}/{data_args.dataset_name}", data_dir=data_args.data_dir)
        features = datasets[list(datasets.keys())[0]].features

    if isinstance(features["mt_tags"].feature, ClassLabel):
        label_list = features["mt_tags"].feature.names
        # No need to convert the labels since they are already ints.
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(datasets["train"]["mt_tags"])
        label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    if data_args.synthetic_train_dir:
        tokenized_synthetic_data = synthetic_train_data.map(
            tokenize_and_align_labels,
            fn_kwargs={
                "tokenizer": tokenizer,
                "padding": padding,
                "label_to_id": label_to_id,
                "label_all_tokens": data_args.label_all_tokens,
                "labels_in_gaps": data_args.dataset_name != "bergamot",
            },
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=False,
        )

    tokenized_datasets = datasets.map(
        tokenize_and_align_labels,
        fn_kwargs={
            "tokenizer": tokenizer,
            "padding": padding,
            "label_to_id": label_to_id,
            "label_all_tokens": data_args.label_all_tokens,
            "labels_in_gaps": data_args.dataset_name != "bergamot",
        },
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=False,
    )

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    if data_args.synthetic_train_dir:
        tokenized_train_data = tokenized_synthetic_data
    else:
        tokenized_train_data = tokenized_datasets["train"] if training_args.do_train else None

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_data,
        eval_dataset=tokenized_datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()  # Saves the tokenizer too for easy upload

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        predict_and_save_output(
            trainer,
            tokenized_eval_dataset=tokenized_datasets["validation"],
            label_list=label_list,
            labels_in_gaps=data_args.dataset_name != "bergamot",
            output_eval_results_file=os.path.join(training_args.output_dir, "valid_results.txt"),
            output_eval_predictions_file=os.path.join(training_args.output_dir, "valid_predictions.txt"),
        )

    if training_args.do_predict:
        logger.info("*** Predict ***")
        predict_and_save_output(
            trainer,
            tokenized_eval_dataset=tokenized_datasets["test"],
            label_list=label_list,
            labels_in_gaps=data_args.dataset_name != "bergamot",
            output_eval_results_file=os.path.join(training_args.output_dir, "test_results.txt"),
            output_eval_predictions_file=os.path.join(training_args.output_dir, "test_predictions.txt"),
        )

    return


def cli_main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    setup_logging(training_args)

    main(model_args, data_args, training_args)


if __name__ == "__main__":
    cli_main()
