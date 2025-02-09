from typing import List
from dataclasses import dataclass
from datasets import load_dataset
from torch.utils.data import Dataset, IterableDataset, DataLoader

import numpy as np
from evaluator import Evaluator

class TaskBase:
    def __init__(self, dataset, data_inputs, data_targets, metrics):
        self.dataset = dataset
        self.data_inputs = data_inputs
        self.data_targets = data_targets

        self.evaluator = Evaluator(metrics)
    
    def get_inputs(self, batch):
        return batch[self.data_inputs]

    def get_targets(self, batch):
        return batch[self.data_targets]

    def to_outputs(self, outputs):
        return outputs

    def _collate_fn(self, batch):
        ret = {}
        for idx, item in enumerate(batch):
            for key, value in item.items():
                if key not in ret:
                    ret[key] = []
                ret[key].append(value)
        return ret

    def get_loader(self, batch_size=1):
        loader = DataLoader(
                self.dataset,
                batch_size=batch_size,
                # shuffle=True,       
                collate_fn=self._collate_fn,
            )
        return loader

    def evaluate(self, predictions: List[str], references: List[str]) -> dict:
        return self.evaluator.evaluate(predictions, references)


class TaskLoader(TaskBase):
    def __init__(self, dataset_path, dataset_name, dataset_split, data_inputs, data_targets, metrics):
        # self.dataset_path = dataset_path
        # self.dataset_name = dataset_name
        # self.dataset_split = dataset_split
        datasets = load_dataset(dataset_path, dataset_name)
        dataset = datasets[dataset_split]
        super().__init__(dataset, data_inputs, data_targets, metrics)


class TaskTruthfulQA(TaskLoader):
    def __init__(self):
        super().__init__(
            dataset_path='truthful_qa',
            dataset_name='generation',
            dataset_split='validation',
            data_inputs='question',
            data_targets='best_answer',
            metrics=['accuracy' 'bleu'],
        )


if __name__ == "__main__":
    task = TaskTruthfulQA()
    loader = task.get_loader(4)
    for idx, batch in enumerate(loader):
        print(idx, batch)