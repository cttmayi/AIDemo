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

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        # metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    #template_format: Optional[str] = field(
    #    default=None,
    #)

    # PEFT Lora
    use_peft_lora: Optional[bool] = field(default=False)
    lora_alpha: Optional[int] = field(default=4)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    lora_target_modules: Optional[str] = field(default= None) # default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",

    # flash attention
    use_flash_attn: Optional[bool] = field(default=False)
    # quantization 4bit
    use_4bit_quantization: Optional[bool] = field(default=False)
    # quantization 8bit
    use_8bit_quantization: Optional[bool] = field(default=False)

@dataclass
class DatasetArguments:
    dataset_name_or_path: Optional[str] = field()
    max_seq_length: Optional[int] = field(default=256)

@dataclass
class TrainArguments(TrainingArguments):
    num_train_epochs: float = field(default=1.0)
    per_device_train_batch_size: int = field(default=4)

    save_strategy: Union[IntervalStrategy, str] = field(default="no")

    evaluation_strategy: Union[IntervalStrategy, str] = field(default="no")
    eval_steps: Optional[float] = field(default=500)
    eval_delay: Optional[float] = field(default=0)

    logging_strategy: Union[IntervalStrategy, str] = field(default="epoch")

    output_dir: str = field(default='./output')
    model_output_dir: str = field(default=None)

    