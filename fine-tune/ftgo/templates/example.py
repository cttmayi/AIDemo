from templates.base import BASE

prompt_key = 'text'
response_key = 'resp'

class SFT(BASE):
    def __init__(self, prompt_max_token=None, response_max_token=None):
        self.prompt_max_token = prompt_max_token
        self.response_max_token = response_max_token


    def preprocess(self, samples):
        batch_prompt = []
        batch_response = []

        for id in range(len(samples[prompt_key])):
            prompt = samples[prompt_key][id]
            response = samples[response_key][id]

            if self.prompt_max_token is not None:
                prompt = self.truncate(prompt, self.prompt_max_token)

            if self.response_max_token is not None:
                prompt = self.truncate(prompt, self.response_max_token)
   
            prompt = '###Qustion:' + prompt + "\n###Anaswer:"
            response = response# + tokenizer.eos_token

            batch_prompt.append(prompt)
            batch_response.append(response)
        return {'prompt': batch_prompt, 'response': batch_response}


class PT(BASE):
    def __init__(self, text_max_token=None):
        self.text_max_token = text_max_token
        pass

    def preprocess(self, samples):
        batch = []
        for id in range(len(samples[prompt_key])):
            text = samples[prompt_key][id]

            if self.text_max_token is not None:
                text = self.truncate(text, self.text_max_token)

            batch.append(text)
        return {'text': batch}