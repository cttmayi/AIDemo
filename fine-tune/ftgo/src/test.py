from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from src.utils.dataset import create_datasets

from rouge import Rouge
from tqdm import tqdm


def create_model(model_name_or_path):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    return model, tokenizer


input_key = "prompt"
label_key = "response"


def generate(model, tokenizer, input_str, config:GenerationConfig=None, device=None):
    input_ids = tokenizer.encode(input_str, return_tensors="pt").to(device)
    output_ids = model.generate(input_ids, config)

    input_str = tokenizer.decode(input_ids[0])
    output_str = tokenizer.decode(output_ids[0][:-1])

    return output_str[len(input_str):]


def process(model_name_or_path, dataset_name_or_path, split='test', max_new_tokens=128, device=None):
    model, tokenizer = create_model(model_name_or_path)
    model.to(device)
    dataset = create_datasets(dataset_name_or_path, split)

    if dataset is None:
        print(f"Dataset({split}) not found.")
    else:
        config = GenerationConfig(
            # temperature=0,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        rouge = Rouge()

        total = 0
        correct = 0

        hyps = []
        refs = []
        errers = []

        for data in tqdm(dataset):
            input = data[input_key]
            label = data[label_key]

            total += 1

            output = generate(model, tokenizer, input, config=config, device=device)
            # print(f"Input: {input}")
            # print(f"Label: {label}")
            # print(f"Output: {output}")
            hyps.append(output)
            refs.append(label)

            score = rouge.get_scores(output, label, avg=True)
            # print(score)

            if label ==  output:
                correct += 1
            else:
                errers.append({'ID:': total, 'output': output, 'label': label})
                pass
        
        scores = rouge.get_scores(hyps, refs, avg=True)
        for key, score in scores.items():
            print(f"{key}: {score}")
        print(f"Accuracy: {int(correct/total*10000)/100}%")
        for errer in errers:
            print(errer)


