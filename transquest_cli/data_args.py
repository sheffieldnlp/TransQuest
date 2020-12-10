from typing import Optional
from dataclasses import dataclass, field


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(default="qe", metadata={"help": "The name of the task."})
    dataset_name: Optional[str] = field(default=None, metadata={"help": "The name of the dataset to use."})
    data_dir: Optional[str] = field(
        default=None, metadata={"help": "The folder with the training, validation and test data for a language pair."},
    )
    synthetic_train_dir: Optional[str] = field(
        default=None, metadata={"help": "The folder with the synthetic data to use for training the model."}
    )
    overwrite_cache: bool = field(default=True, metadata={"help": "Overwrite the cached training and evaluation sets"})
    preprocessing_num_workers: Optional[int] = field(
        default=None, metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
        },
    )
    remove_gaps: bool = field(
        default=False,
        metadata={
            "help": ""
        },
    )

    def __post_init__(self):
        # if self.dataset_name is None and self.train_file is None and self.validation_file is None:
        #     raise ValueError("Need either a dataset name or a training/validation file.")
        # else:
        #     if self.train_file is not None:
        #         extension = self.train_file.split(".")[-1]
        #         assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        #     if self.validation_file is not None:
        #         extension = self.validation_file.split(".")[-1]
        #         assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        self.task_name = self.task_name.lower()

