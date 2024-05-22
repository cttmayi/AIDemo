from utils import config

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import TrainingArguments

from transformers.trainer_utils import (
    EvaluationStrategy,
    FSDPOption,
    HubStrategy,
    IntervalStrategy,
    SchedulerType,
)

# Define and parse arguments.
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    template_format: Optional[str] = field(
        default=None,
        metadata={"help": "Pass None if the dataset is already formatted with the chat template."},
    )

    # PEFT Lora
    use_peft_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT LoRA for training."},
    )
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={"help": "comma separated list of target modules to apply LoRA layers to"},
    )


    # flash attention
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )

    # quantization 4bit
    use_4bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 4bit."},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_storage_dtype: Optional[str] = field(
        default="uint8",
        metadata={"help": "Quantization storage dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    bnb_4bit_use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )

    # quantization 8bit
    use_8bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 8bit."},
    )

    # use_reentrant: Optional[bool] = field(
    #     default=False,
    #     metadata={"help": "Gradient Checkpointing param. Refer the related docs"},
    # )


@dataclass
class DataTrainingArguments:
    dataset_name_or_path: Optional[str] = field(
        # default="timdettmers/openassistant-guanaco",
        metadata={"help": "The preference dataset to use."},
    )

    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )

    dataset_text_field: str = field(
        default="text", metadata={"help": "Dataset field to use as input text."})
    
    max_seq_length: Optional[int] = field(default=config.defualt_model_max_length)

    append_concat_token: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, appends `eos_token_id` at the end of each sample being packed."},
    )

    add_special_tokens: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, tokenizers adds special tokens to each sample being packed."},
    )

@dataclass
class TrainTrainingArguments(TrainingArguments):
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})

    save_strategy: Union[IntervalStrategy, str] = field(
        default="no",
        metadata={"help": "The checkpoint save strategy to use."},
    )

    per_device_train_batch_size: int = field(
        default=4, 
        metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for training."}
    )

    output_dir: str = field(
        # default='output',
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )