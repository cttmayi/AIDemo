import torch
from modelscope import AutoModelForCausalLM, GenerationConfig
from modelscope import AutoTokenizer

from swift import Swift
from swift.llm import get_template, TemplateType, to_device

import cfg

model_dir = 'output/checkpoint-1348'

# 拉起模型
model = AutoModelForCausalLM.from_pretrained(cfg.local_model_path, # torch_dtype=torch.bfloat16,
                                             device_map='auto', trust_remote_code=True)
model = Swift.from_pretrained(model, model_dir)
tokenizer = AutoTokenizer.from_pretrained(cfg.local_model_path, trust_remote_code=True)
template = get_template(TemplateType.default, tokenizer, max_length=1024)

examples, tokenizer_kwargs = template.encode({'query': '好看的'})
if 'input_ids' in examples:
    input_ids = torch.tensor(examples['input_ids'])[None]
    examples['input_ids'] = input_ids
    token_len = input_ids.shape[1]

generation_config = GenerationConfig(
    max_new_tokens=1024,
    temperature=0.3,
    top_k=25,
    top_p=0.8,
    do_sample=True,
    repetition_penalty=1.0,
    # num_beams=10,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id)

device = next(model.parameters()).device
examples = to_device(examples, device)

generate_ids = model.generate(
    generation_config=generation_config,
    **examples)
generate_ids = template.get_generate_ids(generate_ids, token_len)
print(tokenizer.decode(generate_ids, **tokenizer_kwargs))
# I'm an AI language model, so I don't have feelings or physical sensations. However, I'm here to assist you with any questions or tasks you may have. How can I help you today?