from task import TaskBase
from typing import List
from torch.utils.data import Dataset, IterableDataset, DataLoader


class QADataset(Dataset):
    def __init__(self, size=4):

        self.qa = [
            ['中国的首都?', '北京'],
            ['中国的首都是哪里?', '北京'],
            ['中国的首都是哪个城市?', '北京'],
        ]


    def __getitem__(self, index):
        return {
            'question': self.qa[index][0], 
            'response': self.qa[index][1],
        }


    def __len__(self):
        return len(self.qa)


class TaskQA(TaskBase):
    def __init__(self):
        self.dataset = QADataset()
        self.data_inputs = 'question'
        self.data_targets = 'response'
        self.metrics = ["bleu", "llm"]
        super().__init__(self.dataset, self.data_inputs, self.data_targets, self.metrics)