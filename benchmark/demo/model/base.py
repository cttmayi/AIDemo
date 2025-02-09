from typing import Dict, List
import os
from openai import OpenAI


class ModelBase:
    def __init__(self):
        self.name = ''

    def generate(self, prompts: List[str]) -> List[str]:
        raise NotImplementedError


class ModelOpenAI(ModelBase):
    def __init__(self, key, base_url, model):
        
        self.key = key
        self.base_url = base_url
        self.model = model

        self.client = OpenAI(api_key=self.key, base_url=self.base_url)

    def generate(self, prompts: List[str]) -> List[str]:
        if prompts is str:
            prompts = [prompts]

        responses = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            response = self.client.chat.completions.create(
                model = self.model,
                messages = messages
            )
            response = response.choices[0].message.content
            responses.append(response)
        return responses


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)
    model = ModelOpenAI()
    prompts = ["9.11 and 9.8, which is greater?"]
    responses = model.generate(prompts)
    print(responses)