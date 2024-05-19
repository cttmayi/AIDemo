from typing import Dict, List

from datasets import load_dataset
from datasets import Dataset

from transformers import TrainerCallback

def dataset_map(
    dataset: Dataset,
    map_function: callable,
):
    column_names = dataset.column_names
    dataset.map(map_function, batched=False, remove_columns=column_names)


def dataset_split(
    dataset: Dataset,
    size = 0.2,
    seed = 0,
) -> Dict[str, Dataset]:

    test_size = int(size) if size > 1 else size
    dataset = dataset.train_test_split(test_size=test_size, seed=seed)
    return dataset




r = load_dataset('json', data_files='data/text.jsonl')
print(r)