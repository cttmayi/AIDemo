from transformers import AutoTokenizer, AutoModelForCausalLM

from dataclasses import dataclass, field
from transformers import HfArgumentParser

@dataclass
class TestArguments:
    model_name_or_path: str = field(
        metadata={"help": "Model Name"}
    )
    model_max_length: int = field(
        default=512,
        metadata={"help": "Model Max Length"}
    )
    dataset_name_or_path: str = field(
        metadata={"help": "Path to save the data"}
    )
    model_path: str = field(
        metadata={"help": "Path to save the data"}
    )
    device: str = field(
        default='auto',
        metadata={"help": "Device to run the model"}
    )

def create_model(model_name_or_path, device):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return model, tokenizer


def generate(model, tokenizer, input, args:TestArguments):
    input_ids = tokenizer.encode(input, return_tensors="pt", add_special_tokens=False).to(args.device)
    output_ids = model.generate(input_ids, max_length=args.model_max_length)

    return tokenizer.decode(output_ids[0])


if __name__ == "__main__":
    parser = HfArgumentParser((TestArguments,))
    args = parser.parse_args_into_dataclasses()[0]


    model, tokenizer = create_model(args.model_name_or_path, args.device)

    generate(model, tokenizer, "I love this model", args)

