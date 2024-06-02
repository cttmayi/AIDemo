from src import dataset, model, training, test
from src import default
import os

current_dir_path = os.path.dirname(os.path.abspath(__file__))

# dataset
dataset_name_or_path_base = os.path.join(current_dir_path, "data/example.jsonl")
dataset_path_train = "data/example"
dataset_template = 'example'

# model
model_name_or_path_base = "facebook/opt-350m"
model_path_train = "model/opt-350m"
model_max_length = 512
model_prompt_max_length = 512 -32

#train step
# pt
model_path_train_pt = None#model_path_train + "-pt"
num_train_epochs_pt = 2


# sft
model_path_train_sft = model_path_train + "-sft"
num_train_epochs_sft = 100

#train config

import sys

force = True
force_pt = False
force_sft = False

if force or not os.path.exists(dataset_path_train):
    print("dataset process")
    dataset.process(dataset_name_or_path_base, dataset_path_train, model_name_or_path_base, dataset_template)

model_path_train_last = model_name_or_path_base

if model_path_train is not None:
    print("model process")
    if force or not os.path.exists(model_path_train):
        model.process(model_path_train_last, model_path_train)
    model_path_train_last = model_path_train

if model_path_train_pt is not None:
    print("train process pt")
    if force or force_pt or not os.path.exists(model_path_train_pt):
        basic_args = default.BasicArguments(
            dataset_name_or_path=dataset_path_train + '/pt',
            model_name_or_path=model_path_train_last,
            use_peft_lora=True,
            max_seq_length=model_max_length,
        )

        training_args = default.TrainArguments(
            num_train_epochs = num_train_epochs_pt,
            model_output_dir=model_path_train_pt,
        )
        training.process(basic_args, training_args)
    model_path_train_last = model_path_train_pt

if model_path_train_sft is not None:
    print("train process sft")
    if force or force_sft or not os.path.exists(model_path_train_sft):
        basic_args = default.BasicArguments(
            dataset_name_or_path=dataset_path_train + '/sft',
            model_name_or_path=model_path_train_last,
            use_peft_lora=True,
            max_seq_length=model_max_length,
        )
        training_args = default.TrainArguments(
            num_train_epochs = num_train_epochs_sft,
            model_output_dir=model_path_train_sft,
            evaluation_strategy="epoch",
        )
        training.process(basic_args, training_args)
    model_path_train_last = model_path_train_sft


print("test process")
test.process(
    model_name_or_path=model_path_train_last,
    dataset_name_or_path=dataset_path_train + '/sft',
)