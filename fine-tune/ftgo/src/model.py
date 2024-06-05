from transformers import AutoTokenizer, AutoModelForCausalLM

def _save_model(model_name, save_path):
 
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)


def process(name, path_save):
    _save_model(name, path_save)



