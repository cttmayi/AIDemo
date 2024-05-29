from typing import Dict, List

from datasets import load_dataset
from datasets import Dataset
from pathlib import Path

from dataclasses import dataclass, field
from transformers import HfArgumentParser


@dataclass
class DatasetArguments:
    data_path: str = field(
        metadata={"help": "Path to the data"}
        )
    save_path: str = field(
        metadata={"help": "Path to save the data"}
        )
    test_size: float = field(
        default=0.1,
        metadata={"help": "Data size rate for test"}
        )
    save_format: str = field(
        default='jsonl',
        metadata={"help": "Save format(jsonl)"}
    )

def split_data(
    dataset: Dataset,
    size = 0.2,
    seed = 0,
) -> Dict[str, Dataset]:

    test_size = int(size) if size > 1 else size
    dataset = dataset.train_test_split(test_size=test_size, seed=seed)
    return dataset



def load_data(file_path, data_size):
    file_extension = Path(file_path).suffix.lower()
    dataset = None
    if file_extension == '.csv':
        dataset = load_dataset('csv', data_files=file_path)['train']
        dataset = split_data(dataset, data_size)
    elif file_extension == '.json' or file_extension == '.jsonl':
        dataset = load_dataset('json', data_files=file_path)['train']
        dataset = split_data(dataset, data_size)
    else:
        dataset = load_dataset(file_path)
        if len(dataset.keys()) == 1:
            dataset = split_data(dataset['train'], data_size)
        #raise ValueError("Unsupported file format. Please provide a CSV or JSON file.")
        
    return dataset


if __name__ == "__main__":
    parser = HfArgumentParser(DatasetArguments)
    args:DatasetArguments = parser.parse_args_into_dataclasses()[0]
    
    dataset = load_data(args.data_path, args.test_size)
    # splitted_dataset = split_data(dataset)

    print(dataset)

    if args.save_format == 'disk':
        dataset.save_to_disk(args.save_path)
    elif args.save_format == 'jsonl':
        for split_name, split_dataset in dataset.items():
            split_dataset.to_json(f"{args.save_path}/{split_name}.jsonl")

