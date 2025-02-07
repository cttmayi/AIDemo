# https://vllm.hyper.ai/docs/getting-started/quickstart/

from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct")

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
# 打印输出

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


