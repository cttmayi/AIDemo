import torch
from modelscope import AutoModelForCausalLM, GenerationConfig
from modelscope import AutoTokenizer
from modelscope import MsDataset

from swift import Swift
from swift.llm import get_template, TemplateType, to_device

from rouge import Rouge
from tqdm import tqdm

import cfg


final_model_dir = 'output/checkpoint-2000' #cfg.local_model_final_path
original_model_dir = cfg.local_model_path
dataset_path = cfg.local_dataset_test_path


model = AutoModelForCausalLM.from_pretrained(original_model_dir, # torch_dtype=torch.bfloat16,
                                             device_map='auto', trust_remote_code=True)
model = Swift.from_pretrained(model, final_model_dir)
tokenizer = AutoTokenizer.from_pretrained(original_model_dir, trust_remote_code=True)
template = get_template(TemplateType.default, tokenizer, max_length=1024)

device = next(model.parameters()).device

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

def encode(example):
    inst, output = str(example['review']), str(example['label'])
    if output is None:
        return {}

    q = inst
    example, _ = template.encode({'query': q})
    return example

dataset = MsDataset.load('csv', data_files=[dataset_path], split='train').to_hf_dataset()
dataset = dataset.map(encode).filter(lambda e: e.get('input_ids'))

_, tokenizer_kwargs = template.encode({'query': '_'})

data_rouge = Rouge()
hyps = []
refs = []
error_messages = []

for data in tqdm(dataset):
    input_ids = torch.tensor(data['input_ids'])[None]
    token_len = input_ids.shape[1]
    label = str(data['label'])
    query = str(data['review'])

    input_ids = to_device(input_ids, device)

    generate_ids = model.generate(
        generation_config=generation_config,
        input_ids=input_ids,
    )

    generate_ids = template.get_generate_ids(generate_ids, token_len)
    generate_ids = generate_ids[0: -1]

    output = tokenizer.decode(generate_ids, **tokenizer_kwargs)

    hyps.append(output)
    refs.append(label)

    if output != label:
        error_messages.append(f"query: {query}; label: {label}; output: {output}")


scores = data_rouge.get_scores(hyps, refs, avg=True)
for key, score in scores.items():
    # score = int(score * 100)
    print(f"{key}: {score}")
# print(f"Accuracy: {int(correct/total*10000)/100}%")
print('Error messages:')
for error in error_messages:
    print(error)