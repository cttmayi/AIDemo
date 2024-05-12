from typing import Dict, List

from datasets import load_dataset
from datasets import Dataset


def load_dataset():
    pass


def split_dataset(
    dataset: Dataset,
    size = 0.2,
    seed = 0,
) -> Dict[str, Dataset]:

    test_size = int(size) if size > 1 else size
    dataset = dataset.train_test_split(test_size=test_size, seed=seed)
    return dataset


