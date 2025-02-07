

import json
from model import base as model_base
from task import base as task_base

models = [
    model_base.ModelOpenAI,
]

tasks = [
    task_base.TaskTruthfulQA,
]


def save_report(path: str, results):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

def evaluate_once(model: model_base.ModelBase, task: task_base.TaskBase):
    loader = task.get_loader()

    input_all, output_all, target_all = [], [], []
    for batch in loader:
        inputs = task.get_inputs(batch)
        outputs = model.generate(inputs)
        targets = task.get_targets(batch)

        input_all.extend(inputs)
        output_all.extend(outputs)
        target_all.extend(targets)
        break

    metrics = task.evaluate(output_all, target_all)

    return { 
            "task": str(task),
            "metrics": metrics,
            "samples": list(zip(input_all, output_all, target_all))[:3]
        }



if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)

    results = {}
    for task_class in tasks:
        task = task_class()
        for model_class in models:
            model = model_class()
            result = evaluate_once(model, task)

            if str(model) is not results:
                results[str(model)] = []
            results[str(model)].append(result)

    save_report("report.json", results)
            