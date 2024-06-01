import trl
import torch

IGNORE_INDEX = -100

class SFTTrainer(trl.SFTTrainer):

    def _prepare_non_packed_dataloader(self, *args, **kwargs):
        tokenizer = kwargs.get("tokenizer", None)
        dataset = kwargs.get("dataset", None)
        if tokenizer is None:
            tokenizer = args[0]
        if dataset is None:
            dataset = args[1]


        def preprocess_sft_pr(examples):
            model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

            for i in range(len(examples["prompt"])):
                prompt = examples["prompt"][i]
                response = examples["response"][i]
                # message = prompt + response + tokenizer.eos_token

                source_batch = tokenizer(prompt,
                    add_special_tokens=True,
                    truncation=True,
                    padding=False,
                    #max_length=max_seq_length,
                    return_overflowing_tokens=False,
                    return_length=False,
                    return_tensors="pt"
                )
                target_batch = tokenizer(response,
                    add_special_tokens=False,
                    truncation=True,
                    padding=False,
                    #max_length=max_seq_length,
                    return_overflowing_tokens=False,
                    return_length=False,
                    return_tensors="pt"
                )

                source_ids = source_batch["input_ids"][0]
                target_ids = target_batch["input_ids"][0]
                attention_mask = source_batch["attention_mask"][0]
                
                input_ids = torch.cat([source_ids, target_ids, torch.tensor([tokenizer.eos_token_id])], dim=0)
                attention_mask = torch.ones(input_ids.shape, dtype=attention_mask.dtype)
                labels = input_ids.clone()
                labels[:source_ids.shape[0]] = IGNORE_INDEX
                # input_ids[:] = IGNORE_INDEX
                # labels[:] = IGNORE_INDEX
                
                # print('#####')
                # print('s', source_ids, source_ids.dtype)
                # print('t', target_ids, target_ids.dtype)
                # print('i', input_ids, input_ids.dtype)
                # print('l', labels, labels.dtype)
                # print('a', attention_mask, attention_mask.dtype)

                model_inputs["input_ids"].append(input_ids)
                model_inputs["attention_mask"].append(attention_mask)
                model_inputs["labels"].append(labels)

            #print('######')
            #print(model_inputs)

            return model_inputs


        if 'text' in dataset.column_names:
            tokenized_dataset = super()._prepare_non_packed_dataloader(*args, **kwargs)
        else:
            tokenized_dataset = dataset.map(
            preprocess_sft_pr,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=self.dataset_num_proc,
            batch_size=self.dataset_batch_size,
        )

        return tokenized_dataset