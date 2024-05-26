from transformers import AutoTokenizer, AutoModelForCausalLM

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import HfArgumentParser

from utils.dataset import create_datasets
import templates

@dataclass
class TestArguments:
    model_name_or_path: str = field(
        metadata={"help": "Model Name"}
    )
    dataset_name_or_path: str = field(
        metadata={"help": "Path to save the data"}
    )
    template_format: Optional[str] = field(
        default=None,
        metadata={"help": "Pass None if the dataset is already formatted with the chat template."},
    )
    split: str = field(
        default="test",
        metadata={"help": "Split to use"}
    )
    model_max_length: int = field(
        default=512,
        metadata={"help": "Model Max Length"}
    )
    device: str = field(
        default='cpu',
        metadata={"help": "Device to run the model"}
    )

from transformers import OPTForCausalLM, GPT2TokenizerFast
def create_model(model_name_or_path, template_format, device):
    model:OPTForCausalLM = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    template = None
    if template_format is not None:
        template = templates.load_templates(template_format)
        if template is not None:
            tokenizer.chat_template = template.chat_template
        else:
            raise ValueError(f"Template {template_format} not found")

    return model, tokenizer


def generate(model:OPTForCausalLM, tokenizer:GPT2TokenizerFast, input, args:TestArguments):

    input_ids = tokenizer.apply_chat_template(input, return_tensors="pt").to(args.device)
    output_ids = model.generate(input_ids, max_length=args.model_max_length)
    return tokenizer.decode(output_ids[0][len(input_ids[0]):-1])


if __name__ == "__main__":
    parser = HfArgumentParser((TestArguments,))
    args:TestArguments = parser.parse_args_into_dataclasses()[0]

    model, tokenizer = create_model(args.model_name_or_path, args.template_format, args.device)

    dataset = create_datasets(args.dataset_name_or_path, split=args.split)

    total = 0
    correct = 0
    for data in dataset:
        total += 1
        #print(data)
        ret = generate(model, tokenizer, data, args)

        if data["label"] in ret and len(data["label"]) <= len(ret) and len(data["label"]) >= len(ret) - 2:
            correct += 1
        else:
            print('Not correct:', ret, data["label"])
    print(f"Accuracy: {correct/total*100}%")


