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

#train step
# pt
model_path_train_pt = model_path_train + "-pt"
num_train_epochs_pt = 2


# sft
model_path_train_sft = model_path_train + "-sft"
num_train_epochs_sft = 4

#train config

import sys

force = False
force_pt = False
force_sft = True

if True or not os.path.exists(dataset_path_train):
    print("dataset process")
    dataset.process(dataset_name_or_path_base, dataset_path_train, model_name_or_path_base, dataset_template)


if force or not os.path.exists(model_path_train):
    print("model process")
    model.process(model_name_or_path_base, model_path_train)

if force or force_pt or not os.path.exists(model_path_train_pt):
    print("train process pt")
    model_args = default.ModelArguments(
        model_name_or_path=model_path_train,
        use_peft_lora=False
    )
    dataset_args = default.DatasetArguments(
        dataset_name_or_path=dataset_path_train + '/pt',
    )
    training_args = default.TrainArguments(
        num_train_epochs = num_train_epochs_pt,
        model_output_dir=model_path_train_pt,
        evaluation_strategy="epoch",
    )
    training.process(model_args, dataset_args, training_args)

if force or force_sft or not os.path.exists(model_path_train_sft):
    print("train process sft")
    model_args = default.ModelArguments(
        model_name_or_path=model_path_train_pt,
        use_peft_lora=False
    )
    dataset_args = default.DatasetArguments(
        dataset_name_or_path=dataset_path_train + '/sft',
    )
    training_args = default.TrainArguments(
        num_train_epochs = num_train_epochs_sft,
        model_output_dir=model_path_train_sft,
        evaluation_strategy="epoch",
    )
    training.process(model_args, dataset_args, training_args)


print("test process")
test.process(
    model_name_or_path=model_path_train_sft,
    dataset_name_or_path=dataset_path_train + '/sft',
)