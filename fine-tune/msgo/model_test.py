import torch
from modelscope import AutoModelForCausalLM, GenerationConfig
from modelscope import AutoTokenizer
from modelscope import MsDataset


from swift import Swift
from swift.llm import get_template, TemplateType, to_device
# from swift import PeftConfig, PeftModel
from peft import PeftModel, PeftConfig

import cfg

# model_dir = 'output/checkpoint-72'
model_dir = cfg.output_model_path


def init(model_dir, template_type):
    config = PeftConfig.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, model_dir)


    # model = AutoModelForCausalLM.from_pretrained(model_dir, # torch_dtype=torch.bfloat16,
    #                                          device_map='auto', trust_remote_code=True)
    # model = Swift.from_pretrained(model, model_dir)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    template = get_template(template_type, tokenizer, max_length=1024)
    return model, tokenizer, template


def generate(model, tokenizer, generation_config, query):
    examples, tokenizer_kwargs = template.encode({'query': query})

    text = ''
    if 'input_ids' in examples:
        input_ids = torch.tensor(examples['input_ids'])[None]
        examples['input_ids'] = input_ids
        token_len = input_ids.shape[1]

        device = next(model.parameters()).device
        examples = to_device(examples, device)

        generate_ids = model.generate(
            generation_config=generation_config,
            **examples)
        generate_ids = template.get_generate_ids(generate_ids, token_len)[:-1]
        text = tokenizer.decode(generate_ids, **tokenizer_kwargs)
    return text


def encode(example):
    NONE = {'input_ids': [], 'labels': []}

    inp, output = example.get('query', None), example['response']
    if output is None or inp is None:
        return NONE

    example, _ = template.encode({'query': inp, 'response': output})
    # print(example)
    if example.get('input_ids') is None:
        return NONE
    return example


if __name__ == '__main__':
    model, tokenizer, template = init(model_dir, cfg.template_type)


    generation_config = GenerationConfig(
        max_new_tokens=1024,
        temperature=0.0,
        # top_k=25,
        # top_p=0.8,
        do_sample=False,
        repetition_penalty=1.0,
        # num_beams=10,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id)


    dataset = MsDataset.load('json', data_files=[cfg.local_dataset_path_test], split='train').to_hf_dataset()
    dataset = dataset.map(encode)
    dataset = dataset.filter(lambda e: e.get('input_ids') is not None and len(e.get('input_ids')) > 0)
    print(dataset)
    print(dataset[0])

    total = 0
    correct = 0
    for data in dataset:
        total += 1
        query = data["query"]
        ret = generate(model, tokenizer, generation_config, query)

        if data["response"] in ret and len(data["response"]) <= len(ret) and len(data["response"]) >= len(ret) - 2:
            correct += 1
        else:
            print('Not correct:', ret, data["response"])
    print(f"Accuracy: {correct/total*100}%")


    # query = '好看的'

    # text = generate(model, tokenizer, generation_config, query)
    # print(text)