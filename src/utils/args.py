from typing import Optional
from dataclasses import dataclass, field
@dataclass
class ModelArguments:
    """
    Arguments for training model.
    """
    model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the pretrained model."},
    )
    save_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save the fine-tuned model."},
    )
    save_model: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to save the fine-tuned model."},
    )
    do_train: bool = field(
        default=True,
        metadata={"help": "Whether to train on the downstream dataset."},
    )
    do_eval: bool = field(
        default=True,
        metadata={"help": "Whether to evaluate on the downstream dataset."},
    )
    do_test: bool = field(
        default=True,
        metadata={"help": "Whether to test on the downstream dataset."},
    )
    learning_rate: float = field(
        default=None,
        metadata={"help": ""}
    )
    train_batch_size: int = field(
        default=None,
        metadata={"help": ""}
    )
    eval_batch_size: int = field(
        default=None,
        metadata={"help": ""}
    )
    epochs: int = field(
        default=None,
        metadata={"help": ""}
    )
    load_checkpoints: Optional[str] = field(
        default=None,
        metadata={"help": "Path to load the fine-tuned checkpoint."},
    )
    save_results_path: str = field(
        default=None,
        metadata={"help": " Path to save the experimental results: acc/f1micro."}
    )

@dataclass
class DataArguments:
    """
    Arguments for preparing data.
    """

    tokenizer_path: str = field(
        metadata={"help": "Path to tokenizer."}
    )
    cache_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save the train/dev/test data."},
    )
    dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the dataset"},
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the dataset: tabfact/semtabfacts."},
    )
