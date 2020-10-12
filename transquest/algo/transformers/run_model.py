#!/usr/bin/env python
# coding: utf-8


from __future__ import absolute_import, division, print_function

import math
import os
import random
import warnings

import numpy as np
import pandas as pd
import torch

from scipy.stats import mode
from scipy.stats import spearmanr


from sklearn.metrics import (
    matthews_corrcoef,
    confusion_matrix,
    label_ranking_average_precision_score,
    accuracy_score,
)
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm.auto import trange, tqdm

from transformers import AdamW, get_linear_schedule_with_warmup

from transquest.algo.model_classes import model_classes


try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False


class QuestModel:
    def __init__(
        self, model_type, model_name, num_labels=None, weight=None, args=None, use_cuda=True, cuda_device=-1, **kwargs,
    ):

        """
        Initializes a ClassificationModel model.

        Args:
            model_type: The type of model (bert, xlnet, xlm, roberta, distilbert)
            model_name: Default Transformer model name or path to a directory containing Transformer model file (pytorch_nodel.bin).
            num_labels (optional): The number of labels or classes in the dataset.
            weight (optional): A list of length num_labels containing the weights to assign to each label for loss calculation.
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.
        """  # noqa: ignore flake8"

        if args and 'running_seed' in args:
            print('Seed is {}'.format(args['running_seed']))
            random.seed(args['running_seed'])
            np.random.seed(args['running_seed'])
            torch.manual_seed(args['running_seed'])
            if 'n_gpu' in args and args['n_gpu'] > 0:
                torch.cuda.manual_seed_all(args['running_seed'])

        config_class, model_class, tokenizer_class = model_classes[model_type]
        self.config = config_class.from_pretrained(model_name, **args, **kwargs)
        self.num_labels = self.config.num_labels
        self.weight = weight

        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    " Make sure CUDA is available or set use_cuda=False."
                )
        else:
            self.device = "cpu"

        if self.weight:

            self.model = model_class.from_pretrained(
                model_name, config=self.config, weight=torch.Tensor(self.weight).to(self.device), **kwargs)
        else:
            self.model, info = model_class.from_pretrained(model_name, config=self.config, output_loading_info=True, **kwargs)
            print(info)

        self.results = {}

        self.args = {
            "tie_value": 1,
            "stride": 0.8,
            "regression": False,
        }

        if not use_cuda:
            self.args["fp16"] = False

        if args:
            self.args.update(args)

        self.tokenizer = tokenizer_class.from_pretrained(model_name, do_lower_case=self.args["do_lower_case"], **kwargs)

        self.args["model_name"] = model_name
        self.args["model_type"] = model_type

        if model_type in ["camembert", "xlmroberta"]:
            warnings.warn(
                f"use_multiprocessing automatically disabled as {model_type}"
                " fails when using multiprocessing for feature conversion."
            )
            self.args["use_multiprocessing"] = False

        if self.args["wandb_project"] and not wandb_available:
            warnings.warn("wandb_project specified but wandb is not available. Wandb disabled.")
            self.args["wandb_project"] = None

    def train_model(
        self,
        dataset,
        multi_label=False,
        output_dir=None,
        show_running_loss=True,
        args=None,
        eval_df=None,
        verbose=True,
        **kwargs
    ):
        """
        Trains the model using 'train_df'

        Args:
            train_df: Pandas Dataframe containing at least two columns. If the Dataframe has a header, it should contain a 'text' and a 'labels' column. If no header is present,
            the Dataframe should contain at least two columns, with the first column containing the text, and the second column containing the label. The model will be trained on this Dataframe.
            output_dir: The directory where model files will be saved. If not given, self.args['output_dir'] will be used.
            show_running_loss (optional): Set to False to prevent running loss from being printed to console. Defaults to True.
            args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.
            eval_df (optional): A DataFrame against which evaluation will be performed when evaluate_during_training is enabled. Is required if evaluate_during_training is enabled.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

        Returns:
            None
        """  # noqa: ignore flake8"

        if args:
            self.args.update(args)

        if self.args["silent"]:
            show_running_loss = False

        if self.args["evaluate_during_training"] and eval_df is None:
            raise ValueError(
                "evaluate_during_training is enabled but eval_df is not specified."
                " Pass eval_df to model.train_model() if using evaluate_during_training."
            )

        if not output_dir:
            output_dir = self.args["output_dir"]

        if os.path.exists(output_dir) and os.listdir(output_dir) and not self.args["overwrite_output_dir"]:
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Use --overwrite_output_dir to overcome.".format(output_dir)
            )

        self._move_model_to_device()

        os.makedirs(output_dir, exist_ok=True)
        global_step, tr_loss = self.train(
            dataset,
            output_dir,
            multi_label=multi_label,
            show_running_loss=show_running_loss,
            eval_df=eval_df,
            verbose=verbose,
            **kwargs,
        )

        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        if verbose:
            print("Training of {} model complete. Saved to {}.".format(self.args["model_type"], output_dir))

    def train(
        self,
        train_dataset,
        output_dir,
        multi_label=False,
        show_running_loss=True,
        eval_df=None,
        verbose=True,
        **kwargs,
    ):
        """
        Trains the model on train_dataset.

        Utility function to be used by the train_model() method. Not intended to be used directly.
        """

        print('Model will be evaluated on {} examples'.format(len(eval_df)))

        device = self.device
        model = self.model
        args = self.args

        tb_writer = SummaryWriter(logdir=args["tensorboard_dir"])
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args["train_batch_size"])

        t_total = len(train_dataloader) // args["gradient_accumulation_steps"] * args["num_train_epochs"]

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args["weight_decay"],
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        warmup_steps = math.ceil(t_total * args["warmup_ratio"])
        args["warmup_steps"] = warmup_steps if args["warmup_steps"] == 0 else args["warmup_steps"]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args["learning_rate"], eps=args["adam_epsilon"])
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args["warmup_steps"], num_training_steps=t_total
        )

        if args["fp16"]:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

            model, optimizer = amp.initialize(model, optimizer, opt_level=args["fp16_opt_level"])

        if args["n_gpu"] > 1:
            model = torch.nn.DataParallel(model)

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(int(args["num_train_epochs"]), desc="Epoch", disable=args["silent"])
        epoch_number = 0
        best_eval_loss = None
        early_stopping_counter = 0

        if args["evaluate_during_training"]:
            training_progress_scores = self._create_training_progress_scores(multi_label, **kwargs)

        if args["wandb_project"]:
            wandb.init(project=args["wandb_project"], config={**args}, **args["wandb_kwargs"])
            wandb.watch(self.model)

        model.train()
        for _ in train_iterator:
            # epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(tqdm(train_dataloader, desc="Current iteration", disable=args["silent"])):
                batch = tuple(t.to(device) for t in batch)

                inputs = self._get_inputs_dict(batch)
                outputs = model(**inputs)
                # model outputs are always tuple in pytorch-transformers (see doc)
                loss = outputs[0]

                if args["n_gpu"] > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training

                current_loss = loss.item()

                if show_running_loss:
                    print("\rRunning loss: %f" % loss, end="")

                if args["gradient_accumulation_steps"] > 1:
                    loss = loss / args["gradient_accumulation_steps"]

                if args["fp16"]:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    # torch.nn.utils.clip_grad_norm_(
                    #     amp.master_params(optimizer), args["max_grad_norm"]
                    # )
                else:
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(
                    #     model.parameters(), args["max_grad_norm"]
                    # )

                tr_loss += loss.item()
                if (step + 1) % args["gradient_accumulation_steps"] == 0:
                    if args["fp16"]:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args["max_grad_norm"])
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args["max_grad_norm"])

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if args["logging_steps"] > 0 and global_step % args["logging_steps"] == 0:
                        # Log metrics
                        tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args["logging_steps"], global_step)
                        logging_loss = tr_loss
                        if args["wandb_project"]:
                            wandb.log(
                                {
                                    "Training loss": current_loss,
                                    "lr": scheduler.get_lr()[0],
                                    "global_step": global_step,
                                }
                            )

                    if args["save_steps"] > 0 and global_step % args["save_steps"] == 0:
                        # Save model checkpoint
                        output_dir_current = os.path.join(output_dir, "checkpoint-{}".format(global_step))

                        self._save_model(output_dir_current, model=model)

                    if args["evaluate_during_training"] and (
                        args["evaluate_during_training_steps"] > 0
                        and global_step % args["evaluate_during_training_steps"] == 0
                    ):
                        # Only evaluate when single GPU otherwise metrics may not average well
                        results, _ = self.eval_model(
                            eval_df, verbose=verbose and args["evaluate_during_training_verbose"], silent=True, **kwargs
                        )
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)

                        output_dir_current = os.path.join(output_dir, "checkpoint-{}".format(global_step))

                        if args["save_eval_checkpoints"]:
                            self._save_model(output_dir_current, model=model, results=results)

                        training_progress_scores["global_step"].append(global_step)
                        training_progress_scores["train_loss"].append(current_loss)
                        for key in results:
                            training_progress_scores[key].append(results[key])
                        report = pd.DataFrame(training_progress_scores)
                        report.to_csv(
                            os.path.join(args["output_dir"], "training_progress_scores.csv"), index=False,
                        )

                        if args["wandb_project"]:
                            wandb.log(self._get_last_metrics(training_progress_scores))

                        if not best_eval_loss:
                            best_eval_loss = results["eval_loss"]
                            self._save_model(args["best_model_dir"], model=model, results=results)
                        elif results["eval_loss"] - best_eval_loss < args["early_stopping_delta"]:
                            best_eval_loss = results["eval_loss"]
                            self._save_model(args["best_model_dir"], model=model, results=results)
                            early_stopping_counter = 0
                        else:
                            if args["use_early_stopping"]:
                                if early_stopping_counter < args["early_stopping_patience"]:
                                    early_stopping_counter += 1
                                    if verbose:
                                        print()
                                        print(f"No improvement in eval_loss for {early_stopping_counter} steps.")
                                        print(f"Training will stop at {args['early_stopping_patience']} steps.")
                                        print()
                                else:
                                    if verbose:
                                        print()
                                        print(f"Patience of {args['early_stopping_patience']} steps reached.")
                                        print("Training terminated.")
                                        print()
                                    return global_step, tr_loss / global_step

            epoch_number += 1
            output_dir_current = os.path.join(output_dir, "checkpoint-{}-epoch-{}".format(global_step, epoch_number))

            if args["save_model_every_epoch"] or args["evaluate_during_training"]:
                os.makedirs(output_dir_current, exist_ok=True)

            if args["save_model_every_epoch"]:
                self._save_model(output_dir_current, model=model)

            if args["evaluate_during_training"]:
                results, _ = self.eval_model(
                    eval_df, verbose=verbose and args["evaluate_during_training_verbose"], silent=True, **kwargs
                )

                self._save_model(output_dir_current, results=results)

                training_progress_scores["global_step"].append(global_step)
                training_progress_scores["train_loss"].append(current_loss)
                for key in results:
                    training_progress_scores[key].append(results[key])
                report = pd.DataFrame(training_progress_scores)
                report.to_csv(os.path.join(args["output_dir"], "training_progress_scores.csv"), index=False)

                if not best_eval_loss:
                    best_eval_loss = results["eval_loss"]
                    self._save_model(args["best_model_dir"], model=model, results=results)
                elif results["eval_loss"] - best_eval_loss < args["early_stopping_delta"]:
                    best_eval_loss = results["eval_loss"]
                    self._save_model(args["best_model_dir"], model=model, results=results)
                    early_stopping_counter = 0
                else:
                    if args["use_early_stopping"]:
                        if early_stopping_counter < args["early_stopping_patience"]:
                            early_stopping_counter += 1
                            if verbose:
                                print()
                                print(f"No improvement in eval_loss for {early_stopping_counter} steps.")
                                print(f"Training will stop at {args['early_stopping_patience']} steps.")
                                print()
                        else:
                            if verbose:
                                print()
                                print(f"Patience of {args['early_stopping_patience']} steps reached.")
                                print("Training terminated.")
                                print()
                            return global_step, tr_loss / global_step

        return global_step, tr_loss / global_step

    def eval_model(self, dataset, multi_label=False, output_dir=None, verbose=True, silent=False, **kwargs):
        """
        Evaluates the model on eval_df. Saves results to output_dir.

        Args:
            eval_df: Pandas Dataframe containing at least two columns. If the Dataframe has a header, it should contain a 'text' and a 'labels' column. If no header is present,
            the Dataframe should contain at least two columns, with the first column containing the text, and the second column containing the label. The model will be evaluated on this Dataframe.
            output_dir: The directory where model files will be saved. If not given, self.args['output_dir'] will be used.
            verbose: If verbose, results will be printed to the console on completion of evaluation.
            silent: If silent, tqdm progress bars will be hidden.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

        Returns:
            result: Dictionary containing evaluation results. (Matthews correlation coefficient, tp, tn, fp, fn)
            model_outputs: List of model outputs for each row in eval_df
            wrong_preds: List of InputExample objects corresponding to each incorrect prediction by the model
        """  # noqa: ignore flake8"

        if not output_dir:
            output_dir = self.args["output_dir"]

        self._move_model_to_device()

        print('Evaluation set contains {} examples'.format(len(dataset)))

        result, model_outputs = self.evaluate(
            dataset, output_dir, multi_label=multi_label, verbose=verbose, silent=silent, **kwargs
        )
        self.results.update(result)

        if verbose:
            print(self.results)

        return result, model_outputs

    def evaluate(self, dataset, output_dir=None, multi_label=False, prefix="", verbose=True, return_logits=False, silent=False, **kwargs):
        """
        Evaluates the model on eval_df.

        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """

        device = self.device
        model = self.model
        args = self.args
        results = {}

        self._move_model_to_device()

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args["eval_batch_size"])

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        masks = None
        model.eval()

        for batch in tqdm(eval_dataloader, disable=args["silent"] or silent):
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = self._get_inputs_dict(batch)

                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                if not self.args['regression']:
                    logits = logits.sigmoid()
                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
                masks = inputs["attention_mask"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
                masks = np.append(masks, inputs["attention_mask"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps

        if args['regression']:
            preds = np.squeeze(preds)

        if not args['regression']:
            preds = np.argmax(preds, axis=-1)
            if args['word_level']:
                def _remove_padding(a, mask):
                    res = []
                    res_flat = []
                    for i, arr in enumerate(a):
                        res.append(arr[np.nonzero(mask[i])].squeeze())
                        res_flat.extend(arr[np.nonzero(mask[i])].squeeze())
                    return res, res_flat
                preds, preds_flat = _remove_padding(preds, masks)
                out_label_ids, out_label_ids_flat = _remove_padding(out_label_ids, masks)
            else:
                preds_flat = preds
                out_label_ids_flat = out_label_ids
        else:
            preds_flat = preds
            out_label_ids_flat = out_label_ids

        result = self.compute_metrics(preds_flat, out_label_ids_flat, **kwargs)
        result["eval_loss"] = eval_loss
        results.update(result)

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(result.keys()):
                    writer.write("{} = {}\n".format(key, str(result[key])))

        return results, preds

    def compute_metrics(self, preds, labels, multi_label=False, **kwargs):
        """
        Computes the evaluation metrics for the model predictions.

        Args:
            preds: Model predictions
            labels: Ground truth labels
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

        Returns:
            result: Dictionary containing evaluation results. (Matthews correlation coefficient, tp, tn, fp, fn)
            wrong: List of InputExample objects corresponding to each incorrect prediction by the model
        """  # noqa: ignore flake8"

        assert len(preds) == len(labels)

        extra_metrics = {}
        for metric, func in kwargs.items():
            extra_metrics[metric] = func(labels, preds)

        if self.args['regression']:
            return {**extra_metrics}

        mcc = matthews_corrcoef(labels, preds)

        if self.model.num_labels == 2:
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
            return {**{"mcc": mcc, "tp": tp, "tn": tn, "fp": fp, "fn": fn}, **extra_metrics}
        else:
            return {**{"mcc": mcc}, **extra_metrics}

    def predict(self, dataset, multi_label=False):
        """
        Performs predictions on a list of text.

        Args:
            to_predict: A python list of text (str) to be sent to the model for prediction.

        Returns:
            preds: A python list of the predictions (0 or 1) for each text.
            model_outputs: A python list of the raw model outputs for each text.
        """

        device = self.device
        model = self.model
        args = self.args

        self._move_model_to_device()

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args["eval_batch_size"])

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        for batch in tqdm(eval_dataloader, disable=args["silent"]):
            model.eval()
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = self._get_inputs_dict(batch)
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                if multi_label:
                    logits = logits.sigmoid()

                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps

        if not multi_label and args["regression"] is True:
            preds = np.squeeze(preds)
            model_outputs = preds
        else:
            model_outputs = preds
            if multi_label:
                if isinstance(args["threshold"], list):
                    threshold_values = args["threshold"]
                    preds = [
                        [self._threshold(pred, threshold_values[i]) for i, pred in enumerate(example)]
                        for example in preds
                    ]
                else:
                    preds = [[self._threshold(pred, args["threshold"]) for pred in example] for example in preds]
            else:
                preds = np.argmax(preds, axis=1)

        return preds, model_outputs

    def _threshold(self, x, threshold):
        if x >= threshold:
            return 1
        return 0

    def _move_model_to_device(self):
        self.model.to(self.device)

    def _get_inputs_dict(self, batch):
        try:
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], "features_inject": batch[4]}
        except IndexError:
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}

        # XLM, DistilBERT and RoBERTa don't use segment_ids
        if self.args["model_type"] != "distilbert":
            inputs["token_type_ids"] = batch[2] if self.args["model_type"] in ["bert", "xlnet", "albert"] else None

        return inputs

    def _get_last_metrics(self, metric_values):
        return {metric: values[-1] for metric, values in metric_values.items()}

    def _create_training_progress_scores(self, multi_label, **kwargs):
        extra_metrics = {key: [] for key in kwargs}
        if multi_label:
            training_progress_scores = {
                "global_step": [],
                "LRAP": [],
                "train_loss": [],
                "eval_loss": [],
                **extra_metrics,
            }
        else:
            if self.model.num_labels == 2:
                training_progress_scores = {
                    "global_step": [],
                    "tp": [],
                    "tn": [],
                    "fp": [],
                    "fn": [],
                    "mcc": [],
                    "train_loss": [],
                    "eval_loss": [],
                    **extra_metrics,
                }
            elif self.model.num_labels == 1:
                training_progress_scores = {
                    "global_step": [],
                    "train_loss": [],
                    "eval_loss": [],
                    **extra_metrics,
                }
            else:
                training_progress_scores = {
                    "global_step": [],
                    "mcc": [],
                    "train_loss": [],
                    "eval_loss": [],
                    **extra_metrics,
                }

        return training_progress_scores

    def _save_model(self, output_dir, model=None, results=None):
        os.makedirs(output_dir, exist_ok=True)

        if model:
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)

        if results:
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))
