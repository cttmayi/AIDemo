import os, sys
from tqdm import tqdm
import json
from model import ModelBase, ModelOpenAI
from task import TaskBase, TaskMath, TaskQA, TaskLoader, TaskMMLU

def init():
    tasks = [
        ['MMLU', TaskMMLU, [['accuracy', 'bleu']]]
        # ['TruthfulQA', TaskLoader,['truthful_qa', 'generation', 'validation', 'question', 'best_answer', ['accuracy', 'bleu']]],
        # ['Math', TaskMath, []],
        # ['Q&A', TaskQA, []],
    ]
    return tasks




def evaluate_once(task: TaskBase, max_samples=100):
    batch_size = 1
    loader = task.get_loader(batch_size)

    input_all = []
    for batch in loader:
        inputs = task.get_inputs(batch)
        targets = task.get_targets(batch)

        print(f"inputs: {inputs}")
        print(f"targets: {targets}")
        

        input_all.extend(inputs)
        if len(input_all) > max_samples:
            break



if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=False, verbose=True)

    tasks = init()

    results = {}
    for task_class_arg in tasks:
        task_name, task_class, task_args = task_class_arg
        task = task_class(*task_args)
        evaluate_once(task, 3)
            