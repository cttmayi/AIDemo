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

        if compute_dtype == torch.float16 and model_args.use_4bit_quantization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                print("=" * 80)
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
    chat_template = None
    if model_args.use_peft_lora:
        peft_config = LoraConfig(
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            r=model_args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=model_args.lora_target_modules.split(",")
            if model_args.lora_target_modules != "all-linear"
            else model_args.lora_target_modules,
        )

    special_tokens = None
    chat_template = None
    instruction_template = None
    response_template = None

    if model_args.chat_template_format is not None:
        template = templates.load_templates(model_args.chat_template_format)
        special_tokens = template.SpecialTokens
        chat_template = template.DEFAULT_CHAT_TEMPLATE
        instruction_template = template.DEFAULT_INSTRUCTION_TEMPLATE
        response_template = template.DEFAULT_RESPONSE_TEMPLATE 

    # if model_args.chat_template_format == "chatml":
    #     special_tokens = chatml.SpecialTokens
    #     chat_template = chatml.DEFAULT_CHAT_TEMPLATE
    #     instruction_template = chatml.DEFAULT_INSTRUCTION_TEMPLATE
    #     response_template = chatml.DEFAULT_RESPONSE_TEMPLATE        
    # elif model_args.chat_template_format == "zephyr":
    #     special_tokens = zephyr.SpecialTokens
    #     chat_template = zephyr.DEFAULT_CHAT_TEMPLATE
    #     instruction_template = zephyr.DEFAULT_INSTRUCTION_TEMPLATE
    #     response_template = zephyr.DEFAULT_RESPONSE_TEMPLATE

    if special_tokens is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            pad_token=special_tokens.pad_token.value,
            bos_token=special_tokens.bos_token.value,
            eos_token=special_tokens.eos_token.value,
            additional_special_tokens=special_tokens.list(),
            trust_remote_code=True,
        )
        tokenizer.chat_template = chat_template
        # make embedding resizing configurable?
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

    collator = None
    if response_template is not None and instruction_template is not None:
        collator = DataCollatorForCompletionOnlyLM(response_template, instruction_template, okenizer=tokenizer)
    elif response_template is not None:
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    return model, peft_config, tokenizer, collator
