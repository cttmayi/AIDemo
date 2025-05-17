# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional

from swift.llm import DatasetMeta, ResponsePreprocessor, load_dataset, register_dataset

from swift.llm import (InferRequest, Model, ModelGroup, ModelMeta, PtEngine, RequestConfig, TemplateMeta,
                       get_model_tokenizer_with_flash_attn, register_model, register_template)


import os, sys

current_path = os.path.dirname(os.path.abspath(__file__))

class SftPreprocessor(ResponsePreprocessor):
    prompt = """Task: Based on the given two sentences, provide a similarity score between 0.0 and 5.0.
Sentence 1: {text1}
Sentence 2: {text2}
Similarity score: """

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return super().preprocess({
            'query': self.prompt.format(text1=row['text1'], text2=row['text2']),
            'response': f"{row['label']:.1f}"
        })


# register_dataset(
#     DatasetMeta(
#         ms_dataset_id='swift/stsb',
#         # hf_dataset_id='SetFit/stsb',
#         dataset_name='sft_data',
#         preprocess_func=SftPreprocessor(),
#     ))



class PtPreprocessorText(ResponsePreprocessor):
    prompt = """{text}"""

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # text is response
        return super().preprocess({
            'response': self.prompt.format(text=str(row['response'])),
        })


register_dataset(
    DatasetMeta(
        # ms_dataset_id='swift/stsb',
        # hf_dataset_id='SetFit/stsb',
        dataset_path= os.path.join(current_path, 'data', 'pt_text.jsonl'),
        dataset_name='pt_text',
        preprocess_func=PtPreprocessorText(),
    ))

# register_template(
#     TemplateMeta(
#         template_type='custom',
#         prefix=['<extra_id_0>System\n{{SYSTEM}}\n'],
#         prompt=['<extra_id_1>User\n{{QUERY}}\n<extra_id_1>Assistant\n'],
#         chat_sep=['\n']))


# register_model(
#     ModelMeta(
#         model_type='custom',
#         model_groups=[
#             ModelGroup([Model('AI-ModelScope/Nemotron-Mini-4B-Instruct', 'nvidia/Nemotron-Mini-4B-Instruct')])
#         ],
#         template='custom',
#         get_function=get_model_tokenizer_with_flash_attn,
#         ignore_patterns=['nemo']))



if __name__ == '__main__':
    dataset = load_dataset('pt_text')
    # dataset = dataset[0]
    print(f'dataset: {dataset}')
    print(f'dataset[0]: {dataset[0]}')

    # dataset = load_dataset("swift/stsb")[0]
    # print(f'dataset: {dataset}')
    # dataset.to_json(os.path.join(current_path, 'data', 'pt_stsb.jsonl'))
