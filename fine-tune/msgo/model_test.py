from modelscope import GenerationConfig

import cfg
import libs.utils as utils

model_dir = cfg.output_model_path
# model_dir = 'output/checkpoint-1000'
dataset_dir = [cfg.local_dataset_path_test]

if __name__ == '__main__':
    model, tokenizer, template = utils.init_model(model_dir, cfg.template_type)

    dataset = utils.init_dataset(dataset_dir, template)

    generation_config = GenerationConfig(
        max_new_tokens=1024,
        # temperature=0.2,
        # top_k=25,
        # top_p=0.8,
        do_sample=False,
        repetition_penalty=1.0,
        # num_beams=10,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    total = 0
    correct = 0
    for data in dataset:
        total += 1
        query = data["query"]
        ret = utils.generate(model, tokenizer, template, generation_config, query)

        if data["response"] in ret and len(data["response"]) <= len(ret) and len(data["response"]) >= len(ret) - 2:
            correct += 1
        else:
            print('Not correct:', ret, data["response"], f"{correct/total*100}%")
    print(f"Accuracy: {correct/total*100}%")
