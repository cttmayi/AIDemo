from task import TaskBase
from typing import List
from torch.utils.data import Dataset, IterableDataset, DataLoader


class MathDataset(Dataset):
    def __init__(self, size=4):

        self.questions = []
        self.responses = []

        for i in range(size):
            self.questions.append(f"请计算 {2*i}+{2*i+1}, 直接给出结果")
            self.responses.append(f"{4*i+1}")


    def __getitem__(self, index):
        return {
            'question': self.questions[index], 
            'response': self.responses[index]
        }


    def __len__(self):
        return len(self.questions)


class TaskMath(TaskBase):
    def __init__(self):
        self.dataset = MathDataset()
        self.data_inputs = 'question'
        self.data_targets = 'response'
        self.metrics = ["bleu", "llm"]
        super().__init__(self.dataset, self.data_inputs, self.data_targets, self.metrics)

