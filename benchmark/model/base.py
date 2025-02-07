from typing import Dict, List
import os
from openai import OpenAI

class ModelBase:
    def __init__(self, name, key, base_url, model=None):
        self.name = name
        self.key = key
        self.base_url = base_url
        self.model = model if model else self.name

        self.client = OpenAI(api_key=self.key, base_url=self.base_url)

    def __str__(self):
        return self.name

    def generate(self, prompts: List[str]) -> List[str]:
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
        

class ModelOpenAI(ModelBase):
    def __init__(self):
        key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_BASE")
        model = "gpt-3.5-turbo"
        super().__init__('GPT3.5', key, base_url, model)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)
    model = ModelOpenAI()
    prompts = ["9.11 and 9.8, which is greater?"]
    responses = model.generate(prompts)
    print(responses)