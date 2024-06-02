prompt_key = 'text'
response_key = 'resp'

class T:
    def __init__(self, tokenizer, max_token=None):
        self.collator = 'Anaswer:'

        self.tokenizer = tokenizer
        self.max_token = max_token
        self.template_preprocess = {
            'pt': self._pt_preprocess,
            'sft': self._sft_preprocess
        }
        pass

    def _pt_preprocess(self, samples):
        batch = []
        for id in range(len(samples[prompt_key])):
            text = samples[prompt_key][id]
            batch.append(text)
        return {'text': batch}

    def _sft_preprocess(self, samples):
        batch_prompt = []
        batch_response = []

        for id in range(len(samples[prompt_key])):
            prompt = samples[prompt_key][id]
            response = samples[response_key][id]

            prompt = '###Qustion:' + prompt + "\n###Anaswer:"
            response = response# + tokenizer.eos_token

            batch_prompt.append(prompt)
            batch_response.append(response)
        return {'prompt': batch_prompt, 'response': batch_response}




