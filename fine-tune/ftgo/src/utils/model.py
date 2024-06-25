import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import LoraConfig
from src.default import BasicArguments
from peft import PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training


def create_model(model_args:BasicArguments):
    bnb_config = None
    quant_storage_dtype:torch.dtype = None

    if model_args.use_4bit_quantization:
        #compute_dtype = getattr(torch, model_args.bnb_4bit_compute_dtype)
        # quant_storage_dtype = getattr(torch, model_args.bnb_4bit_quant_storage_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=model_args.use_4bit_quantization,
            bnb_4bit_quant_type='nf4', # fp4
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.uint8,
        )
    elif model_args.use_8bit_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=model_args.use_8bit_quantization,
        )

    torch_dtype = torch.float32


    peft_config = None
    if model_args.use_peft_lora:
        peft_config = LoraConfig(
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            r=model_args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules= model_args.lora_target_modules
        )


    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if model_args.use_flash_attn else "eager",
        torch_dtype=torch_dtype, #这个形参可以设置模型中全部 Linear 层的数据格式，torch_dtype=torch.float16 可以使用如下把32位的模型线性层参数转换为16位的模型参数. 但是除线性层之外的所有参数仍为32位浮点数
        # device_map="auto",
    )

    if peft_config:
        # print('load', model_args.model_name_or_path)
        # model = PeftModel(model, peft_config=peft_config)
        try:
            model = PeftModel.from_pretrained(model, model_args.model_name_or_path, is_trainable=True)
            peft_config = None
        except:
            pass

        #model = get_peft_model(model, peft_config)
        # peft_config = None


    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    # print(f"Model: {model_args.model_name_or_path}")
    print('Memory Footprint:', model.get_memory_footprint()/1e9, 'GB')


    return model, peft_config, tokenizer
