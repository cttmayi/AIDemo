from typing import List
from dataclasses import dataclass
from datasets import load_dataset
from torch.utils.data import Dataset, IterableDataset, DataLoader

import numpy as np

@dataclass
class Config:
    task_name: str
    dataset_path: str
    dataset_name: str
    dataset_split: str
    data_inputs: str
    data_targets: str
    metrics: List[str]
    batch_size: int = 4



class Evaluator:
    """评估指标计算中心"""
    @staticmethod
    def evaluate(predictions: List[str], references: List[str], metrics: List[str]) -> dict:
        results = {}
        for metric in metrics:
            if metric == "accuracy":
                results[metric] = np.mean([p == r for p, r in zip(predictions, references)])
            elif metric == "bleu":
                results[metric] = Evaluator._calculate_bleu(predictions, references)
            elif metric == "pass@1":
                results[metric] = Evaluator._calculate_code_pass_rate(predictions, references)
        return results
    
    @staticmethod
    def _calculate_bleu(predictions, references):
        from sacrebleu import corpus_bleu
        return corpus_bleu(predictions, [references]).score
    
    @staticmethod
    def _calculate_code_pass_rate(predictions, references):
        # 这里需要实际执行代码的测试（示例伪代码）
        return 0.0  # 实际应实现代码执行和单元测试验证


class TaskBase:
    def __init__(self, config:Config):
        self.config = config
        datasets = load_dataset(config.dataset_path, config.dataset_name)
        print(datasets)
        self.dataset = datasets[config.dataset_split]

    def __str__(self):
        return self.config.task_name

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
                batch_size=self.config.batch_size,
                collate_fn=self._collate_fn,
            )
        return loader

    def get_inputs(self, batch):
        return batch[self.config.data_inputs]
    
    def get_targets(self, batch):
        return batch[self.config.data_targets]
    
    def evaluate(self, predictions: List[str], references: List[str]) -> dict:
        return Evaluator.evaluate(predictions, references, self.config.metrics)

class TaskTruthfulQA(TaskBase):
    def __init__(self):
        config = Config(
            task_name="TruthfulQA",
            dataset_path="truthful_qa",
            dataset_name="generation",
            dataset_split="validation",
            data_inputs="question",
            data_targets="best_answer",
            metrics=["accuracy", "bleu"],
        )
        super().__init__(config)


if __name__ == "__main__":
    task = TaskTruthfulQA()
    loader = task.get_loader(4)
    for idx, batch in enumerate(loader):
        print(idx, batch)