from enum import Enum
from trl import DataCollatorForCompletionOnlyLM

DEFAULT_INSTRUCTION_TEMPLATE = None
DEFAULT_RESPONSE_TEMPLATE = '\n Answer: '



class SpecialTokens(str, Enum):
    #user = "<|im_start|>user"
    #assistant = "<|im_start|>assistant"
    #system = "<|im_start|>system"
    eos_token = "<|im_end|>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]

class Template:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.special_tokens = SpecialTokens
        self.collator = DataCollatorForCompletionOnlyLM(DEFAULT_RESPONSE_TEMPLATE, tokenizer=tokenizer)

    def preprocess(self, samples):
        batch = []
        
        for id in range(len(samples['text'])):

            text = samples['text'][id]
            label = samples['label'][id]

            text = 'Question: ' + text + DEFAULT_RESPONSE_TEMPLATE + label
            batch.append(text)
        return {'text': batch}





