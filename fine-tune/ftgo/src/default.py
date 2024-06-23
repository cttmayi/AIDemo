from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
#from transformers import TrainingArguments
import transformers 
import torch

from transformers.trainer_utils import (
    EvaluationStrategy,
    FSDPOption,
    HubStrategy,
    IntervalStrategy,
    SchedulerType,
)


@dataclass
class TrainArguments(transformers.TrainingArguments):
    num_train_epochs: float = field(default=1.0)
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=4)

    save_strategy: Union[IntervalStrategy, str] = field(default="no")
    evaluation_strategy: Union[IntervalStrategy, str] = field(default="no")

    eval_delay: Optional[float] = field(default=0)

    logging_strategy: Union[IntervalStrategy, str] = field(default="epoch")

    output_dir: str = field(default='./output')
    

@dataclass
class BasicArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    dataset_name_or_path: Optional[str] = field()

    model_name_or_path: Optional[str] = field()

    max_seq_length: Optional[int] = field(default=256)
    prompt_max_seq_length: Optional[int] = field(default=240)

    # PEFT Lora
    use_peft_lora: Optional[bool] = field(default=False)
    lora_alpha: Optional[int] = field(default=4)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    lora_target_modules: Optional[List] = field(default= None) # default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",

    # flash attention
    use_flash_attn: Optional[bool] = field(default=False)
    # quantization 4bit
    use_4bit_quantization: Optional[bool] = field(default=False)
    # quantization 8bit
    use_8bit_quantization: Optional[bool] = field(default=False)

    model_output_dir: str = field(default=None)

    callbacks: Optional[List] = field(default=None)


    def check_and_fill_args(self, training_args:TrainArguments):
        pass

if torch.backends.mps.is_available():
    default_device = 'mps'
elif torch.cuda.is_available():
    default_device = 'cuda'
else:
    default_device = 'cpu'