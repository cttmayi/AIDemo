from enum import Enum
from trl import DataCollatorForCompletionOnlyLM

DEFAULT_INSTRUCTION_TEMPLATE = None
DEFAULT_RESPONSE_TEMPLATE = '### Output:'


chat_template = "{{'### Instruction: ' + messages['instruction'] + '\n' +  '### Output: '}}"

class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.preprocess = None#DataCollatorForCompletionOnlyLM(DEFAULT_RESPONSE_TEMPLATE, tokenizer=tokenizer)


input_key = 'instruction'
output_key = 'output'

class Template:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        tokenizer.chat_template = chat_template

    def preprocess(self, samples):
        batch = []
        for id in range(len(samples[input_key])):
            input = {input_key: samples[input_key][id]}
            output = samples[output_key][id]
            text = self.tokenizer.apply_chat_template(input, tokenize=False) + output + '\n' + self.tokenizer.eos_token
            batch.append(text)
        return {'text': batch}





