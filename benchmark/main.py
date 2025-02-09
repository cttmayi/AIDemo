import os, sys
from tqdm import tqdm
import json
from model import ModelBase, ModelOpenAI
from task import TaskBase, TaskMath, TaskQA, TaskLoader

def init():
    models = [
        # ['GPT3.5' ,ModelOpenAI, [os.getenv('F2GPT_API_KEY'), os.getenv('F2GPT_API_BASE'), 'gpt-3.5-turbo']],
        ['DEEPSEEK R1 QWEN 1.5B' , ModelOpenAI, [os.getenv('SILICONFLOW_API_KEY'), os.getenv('SILICONFLOW_API_BASE'), 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B']],
        # ['DEEPSEEK R1 QWEN 7B'   , ModelOpenAI, [os.getenv('SILICONFLOW_API_KEY'), os.getenv('SILICONFLOW_API_BASE'), 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B']],
    ]

    tasks = [
        # ['TaskTruthfulQA', task_base.TaskTruthfulQA,[]],
        ['TruthfulQA', TaskLoader,['truthful_qa', 'generation', 'validation', 'question', 'best_answer', ['accuracy', 'bleu']]],
        ['Math', TaskMath, []],
        ['Q&A', TaskQA, []],
    ]
    return models, tasks



def save_report(path: str, results):
    with open(path, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

def evaluate_once(name, model: ModelBase, task: TaskBase, max_samples=100):
    batch_size = 1
    loader = task.get_loader(batch_size)

    input_all, output_all, target_all = [], [], []

    for batch in tqdm(loader, desc=name):
        inputs = task.get_inputs(batch)
        outputs = model.generate(inputs)
        targets = task.get_targets(batch)

        input_all.extend(inputs)
        output_all.extend(outputs)
        target_all.extend(targets)

        if len(input_all) > max_samples:
            break

    metrics = task.evaluate(output_all, target_all)

    return { 
            "metrics": metrics,
            "samples": list(zip(input_all, output_all, target_all))[:3]
        }


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=False, verbose=True)

    models, tasks = init()

    results = {}
    for task_class_arg in tasks:
        task_name, task_class, task_args = task_class_arg
        task = task_class(*task_args)
        for model_class_arg in models:
            model_name, model_class, model_args = model_class_arg
            # print(model_class, *model_args)
            model = model_class(*model_args)
            result = evaluate_once(model_name + " - " + task_name, model, task)

            if model_name not in results:
                results[model_name] = {}
            results[model_name][task_name] = result
    print(results)
    save_report("report.json", results)
            