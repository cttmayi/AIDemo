import datasets
import templates
from pathlib import Path
import os
from datasets import Dataset 

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

import json

def _load_directory(dir_path):
    json_lines = []
    for file_name in Path(dir_path).iterdir():
        if file_name.is_file():
            content = file_name.read_text()
            json_lines.append({'text': content})
    return json_lines

def lines_to_json(json_lines, path_save):
    split_name = 'train'
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    
    with open(f"{path_save}/{split_name}.jsonl", "w") as f:
        for json_line in json_lines:
            f.write(json.dumps(json_line))
            f.write('\n')


def _load_dataset(file_path, test_size):
    file_extension = Path(file_path).suffix.lower()
    dataset = None

    if file_extension == '.csv':
        dataset = datasets.load_dataset('csv', data_files=file_path)
    elif file_extension == '.json' or file_extension == '.jsonl':
        dataset = datasets.load_dataset('json', data_files=file_path)
    else:
        dataset = datasets.load_dataset(file_path)

    if len(dataset.keys()) == 1:
        if test_size != 0:
            dataset = dataset['train'].train_test_split(test_size=test_size, seed=0)
    else:
        print("Dataset has multiple splits, ignore test_size argument")
        
    return dataset


def process(path:str, path_save:str, model_name_or_path, template_format=None, test_size=0):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    template_preprocess = {}
    if template_format is not None:
        template_file = templates.load_templates(template_format)
        if template_file is not None:
            template = template_file.T(tokenizer, 512)
            # print(template)
            template_preprocess = template.template_preprocess
        else:
            raise ValueError(f"Template {template_format} not found")

    is_directory = Path(path).is_dir()

    if is_directory:
        json_lines = _load_directory(path)
        lines_to_json(json_lines, path_save)

    else:
        dataset = _load_dataset(path, test_size)
        for split_name, split_dataset in dataset.items():
            split_dataset:Dataset
            split_dataset = split_dataset.map(template_preprocess, 
                                                batched=True, 
                                                remove_columns=split_dataset.column_names, 
                                                load_from_cache_file=False
                                                )
            split_dataset.to_json(f"{path_save}/{split_name}.jsonl", force_ascii=False)






    


