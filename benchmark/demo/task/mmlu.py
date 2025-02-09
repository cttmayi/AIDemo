
from task import TaskBase


from datasets import load_dataset

class TaskMMLU(TaskBase):
    def __init__(self, metrics):
        dataset_path = 'cais/mmlu'
        dataset_name = 'abstract_algebra'
        dataset_split = 'test'
        data_inputs = 'question'
        data_targets = 'answer'
        datasets = load_dataset(dataset_path, dataset_name)
        print(datasets)
        dataset = datasets[dataset_split]
        super().__init__(dataset, data_inputs, data_targets, metrics)

        self._ABCD = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

    def _template(self, question, choices):
        choice_ABCD = []
        for idx, choice in enumerate(choices):
            # print(choice, type(choice))
            choice = self._ABCD[idx] + '. ' + str(choice) + ';'
            choice_ABCD.append(choice)
        choice_ABCD = ' '.join(choice_ABCD)

        return f'{question}\n{choice_ABCD}\n'
    
    def _extract_boxed_content(self, latex_string):
        import re
        # 使用正则表达式匹配 \boxed{} 中的内容
        pattern = r"\\boxed\{([^}]*)\}"
        matches = re.findall(pattern, latex_string)
        if len(matches) > 0:
            return matches[0]
        return latex_string

    def to_outputs(self, outputs):
        ret = []
        for idx, output in enumerate(outputs):
            output = self._extract_boxed_content(output)
            ret.append(output)
        return ret
    
    def _collate_fn(self, batch):
        ret = {
            'question': [],
            'choice': [],
            'answer': []
        }
        for idx, item in enumerate(batch):
            question = item['question']
            choices = item['choices']
            answer = item['answer']

            ret['question'].append(self._template(question, choices))
            ret['answer'].append(self._ABCD[int(answer)])
        return ret


        