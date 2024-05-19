
from trl.commands.cli_utils import SftScriptArguments, TrlParser


from transformers import TrainingArguments

from trl import (
    ModelConfig,
    get_peft_config, 
)

# python examples/scripts/sft.py --output_dir sft_openassistant-guanaco  --model_name meta-llama/Llama-2-7b-hf --dataset_name timdettmers/openassistant-guanaco --load_in_4bit --use_peft --per_device_train_batch_size 4 --gradient_accumulation_steps 2
if __name__ == "__main__":
    parser = TrlParser((SftScriptArguments, TrainingArguments, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()

    peft_config=get_peft_config(model_config)

    print('args', args)
    print('training_args', training_args)
    print('model_config', model_config)
    print('peft_config', peft_config)