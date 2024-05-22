from typing import Dict, List

from datasets import load_dataset
from datasets import Dataset
from pathlib import Path

from dataclasses import dataclass, field
from transformers import HfArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM

@dataclass
class ModelArguments:
    model_name: str = field(
        metadata={"help": "Model Name"}
        )
    model_path: str = field(
        metadata={"help": "Path to save the data"}
        )


def save_model(model_name, save_path):


    model = AutoModelForCausalLM.from_pretrained(model_name).to("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)


if __name__ == "__main__":
    parser = HfArgumentParser(ModelArguments)
    args:ModelArguments = parser.parse_args_into_dataclasses()[0]
    
    save_model(args.model_name, args.model_path)


