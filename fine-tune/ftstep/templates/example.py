from enum import Enum
from trl import DataCollatorForCompletionOnlyLM

DEFAULT_INSTRUCTION_TEMPLATE = None
DEFAULT_RESPONSE_TEMPLATE = '\n Answer: '

_tokenizer = None

class _SpecialTokens(str, Enum):
    user = "<|im_start|>user"
    assistant = "<|im_start|>assistant"
    system = "<|im_start|>system"
    eos_token = "<|im_end|>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]

def _preprocess(samples):
    batch = []
    
    for id in range(len(samples['text'])):

        text = samples['text'][id]
        label = samples['label'][id]

        text = 'Question: ' + text + DEFAULT_RESPONSE_TEMPLATE + label
        batch.append(text)
    return {'text': batch}


def set_tokenizer(tokenizer):
    global _tokenizer
    _tokenizer = tokenizer

def get_collator():
    collator = DataCollatorForCompletionOnlyLM(DEFAULT_RESPONSE_TEMPLATE, tokenizer=_tokenizer)
    return collator

def get_special_tokens():
    return _SpecialTokens

def get_dataset_preprocess():
    return _preprocess

