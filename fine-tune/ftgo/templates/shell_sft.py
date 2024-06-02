prompt_key = 'instruction'
response_key = 'cmd'

class T:
    def __init__(self, tokenizer, max_token=None):
        self.collator = 'Anaswer:'

        self.tokenizer = tokenizer
        self.max_token = max_token
        self.template_preprocess = self._sft_preprocess

    def _sft_preprocess(self, samples):
        batch_prompt = []
        batch_response = []

        for id in range(len(samples[prompt_key])):
            prompt = samples[prompt_key][id]
            response = samples[response_key][id]

            prompt = '###Instruction:' + prompt + "\n###Shell:"
            response = response# + tokenizer.eos_token

            batch_prompt.append(prompt)
            batch_response.append(response)
        return {'prompt': batch_prompt, 'response': batch_response}




