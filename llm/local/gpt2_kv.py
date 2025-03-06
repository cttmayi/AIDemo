import numpy as np  
import time  
import torch  
from transformers import AutoModelForCausalLM, AutoTokenizer  

model_name = "gpt2"
model_name = "Qwen/Qwen2-0.5B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"  
tokenizer = AutoTokenizer.from_pretrained(model_name)  
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)  
  
for use_cache in (True, False):  
    times = []  
    for _ in range(1): # measuring 10 generations  
        start = time.time()  
        model.generate(**tokenizer("What is KV caching?", return_tensors="pt").to(device), use_cache=use_cache, max_new_tokens=100)  
        times.append(time.time() - start)  
    print(f"{'with' if use_cache else 'without'} KV caching: {round(np.mean(times), 3)} +- {round(np.std(times), 3)} seconds")