
import os

from datasets import DatasetDict, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError
from datasets.exceptions import DatasetNotFoundError

from utils.argument import DataTrainingArguments

def create_datasets(dataset_name_or_path, dataset_preprocess=None, split='train'):
    try:
        # Try first if dataset on a Hub repo
        dataset = load_dataset(dataset_name_or_path, split=split)
    except (DatasetNotFoundError, DatasetGenerationError):
        # If not, check local dataset
        dataset = load_from_disk(os.path.join(dataset_name_or_path, split=split))

    if dataset_preprocess:
        print('doing preprocessing...')
        dataset = dataset.map(
            dataset_preprocess,
            batched=True,
            remove_columns=dataset.column_names,
        )

    print(f"Size of the {split} set: {len(dataset)}.")
    print(f"A sample of {split} dataset: {dataset[0]}")

    return dataset