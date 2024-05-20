
import os

from datasets import DatasetDict, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError
from datasets.exceptions import DatasetNotFoundError

from utils.argument import DataTrainingArguments

def create_datasets(tokenizer, data_args:DataTrainingArguments, dataset_preprocess=None):

    try:
        # Try first if dataset on a Hub repo
        dataset = load_dataset(data_args.dataset_name, split='train')
    except (DatasetNotFoundError, DatasetGenerationError):
        # If not, check local dataset
        dataset = load_from_disk(os.path.join(data_args.dataset_name, split='train'))

    if dataset_preprocess:
        dataset = dataset.map(
            dataset_preprocess,
            batched=True,
            remove_columns=dataset.column_names,
        )

    print(f"Size of the train set: {len(dataset)}.")
    print(f"A sample of train dataset: {dataset[0]}")

    return dataset