import numpy as np
import os
import torch

from torch.utils.data import TensorDataset

from transquest.data.collate import convert_examples_to_features


def make_dataset(
    examples, tokenizer, config, evaluate=False, no_cache=False, multi_label=False, verbose=True, silent=False,
):
    """
    Converts a list of InputExample objects to a TensorDataset containing InputFeatures. Caches the InputFeatures.
    """

    process_count = config["process_count"]

    if not no_cache:
        no_cache = config["no_cache"]

    if not multi_label and config["regression"]:
        output_mode = "regression"
    else:
        output_mode = "classification"

    os.makedirs(config["cache_dir"], exist_ok=True)

    mode = "dev" if evaluate else "train"
    cached_features_file = os.path.join(
        config["cache_dir"],
        "cached_{}_{}_{}_{}_{}".format(
            mode, config["model_type"], config["max_seq_length"], config["num_labels"], len(examples),
        ),
    )

    if os.path.exists(cached_features_file) and (
            (not config["reprocess_input_data"] and not no_cache) or (
            mode == "dev" and config["use_cached_eval_features"] and not no_cache)
    ):
        features = torch.load(cached_features_file)
        if verbose:
            print(f"Features loaded from cache at {cached_features_file}")
    else:
        if verbose:
            print(f"Converting to features started. Cache is not used.")
            if config["sliding_window"]:
                print("Sliding window enabled")
        features = convert_examples_to_features(
            examples,
            config["max_seq_length"],
            tokenizer,
            output_mode,
            # XLNet has a CLS token at the end
            cls_token_at_end=bool(config["model_type"] in ["xlnet"]),
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if config["model_type"] in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            # RoBERTa uses an extra separator b/w pairs of sentences,
            # cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            sep_token_extra=bool(config["model_type"] in ["roberta", "camembert", "xlmroberta"]),
            # PAD on the left for XLNet
            pad_on_left=bool(config["model_type"] in ["xlnet"]),
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if config["model_type"] in ["xlnet"] else 0,
            process_count=process_count,
            multi_label=multi_label,
            silent=config["silent"] or silent,
            use_multiprocessing=config["use_multiprocessing"],
            sliding_window=config["sliding_window"],
            flatten=not evaluate,
            stride=config["stride"],
        )
        if verbose and config["sliding_window"]:
            print(f"{len(features)} features created from {len(examples)} samples.")

        if not no_cache:
            torch.save(features, cached_features_file)

    if config["sliding_window"] and evaluate:
        window_counts = [len(sample) for sample in features]
        features = [feature for feature_set in features for feature in feature_set]

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_features = None
    if features[0].features_inject:
        num_features = len(features[0].features_inject)
        features_arr = np.zeros((len(features), num_features))
        for i, f in enumerate(features):
            for j, feature_name in enumerate(f.features_inject.keys()):
                features_arr[i][j] = f.features_inject[feature_name]
        all_features = torch.tensor(features_arr, dtype=torch.float)

    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    if all_features is not None:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_features)
    else:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    if config["sliding_window"] and evaluate:
        return dataset, window_counts
    else:
        return dataset
