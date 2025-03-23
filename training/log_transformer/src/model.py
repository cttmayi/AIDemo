from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    Qwen2TokenizerFast, Qwen2ForCausalLM, Qwen2Config
)


def create_model(base_model_path, **kwargs):
    config:Qwen2Config = AutoConfig.from_pretrained(base_model_path)
    config.num_attention_heads = kwargs.get('num_attention_heads', 4) # 14
    config.num_key_value_heads = kwargs.get('num_key_value_heads', 2) # 2
    config.hidden_size = kwargs.get('hidden_size', 128) # 896
    config.num_hidden_layers = kwargs.get('num_hidden_layers', 6) # 24

    model:Qwen2ForCausalLM = AutoModelForCausalLM.from_config(config)

    # # 计算参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {int(num_params)/1000_000}M")

    return model