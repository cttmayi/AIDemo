
import os

from datasets import DatasetDict, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError
from datasets.exceptions import DatasetNotFoundError

def create_datasets(dataset_name_or_path, split='train'):
    try:
        # Try first if dataset on a Hub repo
        dataset = load_dataset(dataset_name_or_path, split=split)
    except (DatasetNotFoundError, DatasetGenerationError):
        # If not, check local dataset
        dataset = load_from_disk(dataset_name_or_path, split=split)

    print(f"Size of the {split} set: {len(dataset)}.")
    print(f"A sample of {split} dataset: {dataset[0]}")

    return dataset