import datasets
import templates
from pathlib import Path

from datasets import Dataset 

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def _load_dataset(file_path, test_size):
    file_extension = Path(file_path).suffix.lower()
    dataset = None
    if file_extension == '.csv':
        dataset = datasets.load_dataset('csv', data_files=file_path)['train']
        dataset = dataset.train_test_split(test_size=test_size, seed=0)
    elif file_extension == '.json' or file_extension == '.jsonl':
        dataset = datasets.load_dataset('json', data_files=file_path)['train']
        dataset = dataset.train_test_split(test_size=test_size, seed=0)
    else:
        dataset = datasets.load_dataset(file_path)
        if len(dataset.keys()) == 1:
            dataset = dataset['train'].train_test_split(test_size=test_size, seed=0)
        
    return dataset


def process(path:str, path_save:str, model_name_or_path, template_format=None, test_size=0.1):

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    template_preprocess = {}
    if template_format is not None:
        template_file = templates.load_templates(template_format)
        if template_file is not None:
            template = template_file.T(tokenizer, 512)
            print(template)
            template_preprocess = template.template_preprocess
        else:
            raise ValueError(f"Template {template_format} not found")

    dataset = _load_dataset(path, test_size)

    for preprocess_name, preprocess_function in template_preprocess.items():
        for split_name, split_dataset in dataset.items():
            split_dataset:Dataset
            split_dataset = split_dataset.map(preprocess_function, 
                                              batched=True, 
                                              remove_columns=split_dataset.column_names, 
                                              load_from_cache_file=False
                                              )
            split_dataset.to_json(f"{path_save}/{preprocess_name}/{split_name}.jsonl")

            #print(f"Size of the {split_name} set: {len(split_dataset)}.")
            #print(f"A sample of {split_name} dataset: {split_dataset[0]}")

            # if preprocess_name == 'sft':
            #     example = split_dataset[0]
            #     print('prompt:')
            #     print(tokenizer.encode(example['prompt']))
            #     print(tokenizer.tokenize(example['prompt']))
            #     print('response:')
            #     print(tokenizer.encode(example['response']))
            #     print(tokenizer.tokenize(example['response']))
            #     print('prompt + response:')
            #     print(tokenizer.encode(example['prompt']+example['response']))
            #     print(tokenizer.tokenize(example['prompt']+example['response']))




    


