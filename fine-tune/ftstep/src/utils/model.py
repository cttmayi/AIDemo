import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import DataCollatorForCompletionOnlyLM

from peft import LoraConfig
from utils.argument import ModelArguments

import templates

def create_model(model_args:ModelArguments):
    bnb_config = None
    quant_storage_dtype = None

    if model_args.use_4bit_quantization:
        compute_dtype = getattr(torch, model_args.bnb_4bit_compute_dtype)
        quant_storage_dtype = getattr(torch, model_args.bnb_4bit_quant_storage_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=model_args.use_4bit_quantization,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=model_args.bnb_4bit_use_nested_quant,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )
    elif model_args.use_8bit_quantization:
        bnb_config = BitsAndBytesConfig(load_in_8bit=model_args.use_8bit_quantization)

    torch_dtype = (
        quant_storage_dtype if quant_storage_dtype and quant_storage_dtype.is_floating_point else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if model_args.use_flash_attn else "eager",
        torch_dtype=torch_dtype,
    )

    peft_config = None

    if model_args.use_peft_lora:
        peft_config = LoraConfig(
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            r=model_args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules= model_args.lora_target_modules.split(",") if model_args.lora_target_modules else None,
            #if model_args.lora_target_modules != "all-linear"
            #else model_args.lora_target_modules,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    # tokenizer.pad_token = tokenizer.eos_token

    preprocess = None
    collator = None

    template = None
    if model_args.template_format is not None:
        template = templates.load_templates(model_args.template_format)
        if template is not None:
            preprocess = template.Template(tokenizer).preprocess
            collator = template.Collator(tokenizer).preprocess
            # tokenizer.chat_template = template.chat_template
        else:
            raise ValueError(f"Template {model_args.template_format} not found")

    return model, peft_config, tokenizer, collator, preprocess
