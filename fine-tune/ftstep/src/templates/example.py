from enum import Enum
from trl import DataCollatorForCompletionOnlyLM

DEFAULT_INSTRUCTION_TEMPLATE = None
DEFAULT_RESPONSE_TEMPLATE = '### Answer:'


chat_template = "{{'### Question: ' + messages['text'] + '\n' +  '### Answer: '}}"

class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.preprocess = None#DataCollatorForCompletionOnlyLM(DEFAULT_RESPONSE_TEMPLATE, tokenizer=tokenizer)

class Template:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        tokenizer.chat_template = chat_template

    def preprocess(self, samples):
        batch = []
        for id in range(len(samples['text'])):
            text = samples['text'][id]
            label = samples['label'][id]
            input = {'text': text}
            text = self.tokenizer.apply_chat_template(input, tokenize=False) + label + '\n' + self.tokenizer.eos_token
            print(text)
            batch.append(text)
        return {'text': batch}





