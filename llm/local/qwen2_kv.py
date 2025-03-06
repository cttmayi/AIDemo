from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import time

device = "cpu" # the device to load the model onto

model_name = "Qwen/Qwen2-0.5B-Instruct"
model_local_path = "models/Qwen2-0.5B-Instruct"


if os.path.exists(model_local_path):
    model_name = model_local_path

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map=device,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

if model_name != model_local_path:
    model.save_pretrained(model_local_path)
    tokenizer.save_pretrained(model_local_path)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print('text=====')
print(text)
print('=========')


model_inputs = tokenizer([text], return_tensors="pt").to(device)

def generate(model, model_inputs, use_cache=True, past_key_values=None):
    start_time = time.time()  # 获取开始时间
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        do_sample=False,
        use_cache=use_cache,
        past_key_values = past_key_values,
        k = 0
    )
    end_time = time.time()  # 获取结束时间
    print(f"cache:{use_cache}, {past_key_values is not None} __ 时间：{end_time - start_time:.4f} 秒")
    return generated_ids


generated_ids = generate(model, model_inputs)
generated_ids = generate(model, model_inputs, use_cache=False, past_key_values=generated_ids.past_key_values)


generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print('response=====')
print(response)