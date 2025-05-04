import torch
from modelscope import AutoModelForCausalLM
from modelscope import AutoTokenizer
from modelscope import MsDataset

from swift import Swift
from swift.llm import get_template, TemplateType, to_device
# from swift import PeftConfig, PeftModel
from peft import PeftModel, PeftConfig
from swift import Swift, LoraConfig, PromptEncoderConfig

MAX_LENGYH = 256

def create_model(model_type, template_type):
    print('Loading model:', model_type)
    model = AutoModelForCausalLM.from_pretrained(model_type, device_map='auto')
    peft_config = LoraConfig(
                    r=8,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    lora_alpha=32,
                    lora_dropout=0.05)


    model = Swift.prepare_model(model, peft_config)
    model.print_trainable_parameters()

    template_type = template_type or model.model_meta.template
    print('Loading template:', template_type)

    tokenizer = AutoTokenizer.from_pretrained(model_type, trust_remote_code=True)
    template = get_template(template_type, tokenizer, max_length=MAX_LENGYH)

    return model, tokenizer, template


def init_model(model_dir, template_type):
    config = PeftConfig.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map='auto')
    model = PeftModel.from_pretrained(model, model_dir)
    model.eval()

    # model = AutoModelForCausalLM.from_pretrained(model_dir, # torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
    # model = Swift.from_pretrained(model, model_dir)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    template = get_template(template_type, tokenizer, max_length=MAX_LENGYH)
    return model, tokenizer, template


def generate(model, tokenizer, template, generation_config, query):
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


def init_dataset(dataset_dir, template):
    NONE = {'input_ids': [], 'labels': []}

    def encode(example):
        inp, output = example.get('query', None), example['response']
        if output is None or inp is None:
            return NONE

        example, _ = template.encode({'query': inp, 'response': output})
        # print(example)
        if example.get('input_ids') is None:
            return NONE
        return example

    dataset = MsDataset.load('json', data_files=dataset_dir, split='train').to_hf_dataset()
    dataset = dataset.map(encode)
    dataset = dataset.filter(lambda e: e.get('input_ids') is not None and len(e.get('input_ids')) > 0)

    print(dataset)
    print(dataset[0])

    return dataset