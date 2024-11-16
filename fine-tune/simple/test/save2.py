
model_id = 'qwen/Qwen2-0.5B'
model_dir = "_model/" + model_id
model_dir_2 = "_model2/" + model_id
# name = "qwen/Qwen-7B-Chat-Int4"

from modelscope import AutoTokenizer, AutoModelForCausalLM, snapshot_download

model_dir = snapshot_download(model_id, local_dir=model_dir)


tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
model.save_pretrained(model_dir_2)
tokenizer.save_pretrained(model_dir_2)
print("model saved to _model/" + model_id)
