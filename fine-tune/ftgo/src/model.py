from transformers import AutoTokenizer, AutoModelForCausalLM

def process(model_name, save_path, cache_dir=None):
 
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir).to("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
