from src import dataset, model, training, test
from src import default
import os

NO = 0
YES = 1
FORCE = 2

current_dir_path = os.path.dirname(os.path.abspath(__file__))

# MODEL =================
model_name_or_path_base = "facebook/opt-350m"
model_path_train = "model/opt-350m"


# FINETUNE PF ===========
dataset_name_or_path_pt_base = os.path.join(current_dir_path, "data/example.jsonl")
dataset_path_train_pt = "data/example/pt"
dataset_template_pt = 'example_pt'
dataset_test_data_size_pt = 0

model_path_train_pt = model_path_train + "-pt"
num_train_epochs_pt = 2

# FINETUNE SFT ==========
dataset_name_or_path_sft_base = os.path.join(current_dir_path, "data/example.jsonl")
dataset_path_train_sft = "data/example/sft"
dataset_template_sft = 'example_sft'
dataset_test_data_size_sft = 0.1

model_path_train_sft = model_path_train + "-sft"
num_train_epochs_sft = 10

# TRAIN Config ===========
train_use_peft_lora = False
train_batch_size = 4
train_max_length = 512

# TEST Config ============
test_max_new_tokens = 16

# Workflow Config: NO; YES; FORCE
is_dataset_pt = YES
is_dataset_sft = YES
is_finetune_pt = NO
is_finetune_sft = YES



# =====================================
if is_dataset_pt != NO:
    print("dataset process for pt")
    if is_dataset_pt == FORCE or not os.path.exists(dataset_path_train_pt):
        dataset.process(dataset_name_or_path_pt_base, dataset_path_train_pt, model_name_or_path_base, dataset_template_pt, dataset_test_data_size_pt)

if is_dataset_sft != NO:
    print("dataset process for sft")
    if is_dataset_sft == FORCE or not os.path.exists(dataset_path_train_sft):
        dataset.process(dataset_name_or_path_sft_base, dataset_path_train_sft, model_name_or_path_base, dataset_template_sft, dataset_test_data_size_sft)


model_path_train_last = model_name_or_path_base

if model_path_train is not None:
    print("model process")
    if not os.path.exists(model_path_train):
        model.process(model_path_train_last, model_path_train)
    model_path_train_last = model_path_train

if is_finetune_pt != NO:
    print("train process pt")
    if is_finetune_pt == FORCE or not os.path.exists(model_path_train_pt):
        basic_args = default.BasicArguments(
            dataset_name_or_path=dataset_path_train_pt,
            model_name_or_path=model_path_train_last,
            use_peft_lora=train_use_peft_lora,
            max_seq_length=train_max_length,
        )
        training_args = default.TrainArguments(
            num_train_epochs = num_train_epochs_pt,
            model_output_dir=model_path_train_pt,
            per_device_eval_batch_size=train_batch_size,
            per_device_train_batch_size=train_batch_size,
        )
        training.process(basic_args, training_args)
    model_path_train_last = model_path_train_pt

if is_finetune_sft != NO:
    print("train process sft")
    if is_finetune_sft == FORCE or not os.path.exists(model_path_train_sft):
        basic_args = default.BasicArguments(
            dataset_name_or_path=dataset_path_train_sft,
            model_name_or_path=model_path_train_last,
            use_peft_lora=train_use_peft_lora,
            max_seq_length=train_max_length,
        )
        training_args = default.TrainArguments(
            num_train_epochs=num_train_epochs_sft,
            model_output_dir=model_path_train_sft,
            evaluation_strategy="epoch",
            per_device_eval_batch_size=train_batch_size,
            per_device_train_batch_size=train_batch_size,
        )
        training.process(basic_args, training_args)
    model_path_train_last = model_path_train_sft


print("test process for test dataset")
test.process(
    model_name_or_path=model_path_train_last,
    dataset_name_or_path=dataset_path_train_sft,
    split='test',
    max_new_tokens=test_max_new_tokens,
)
print("test process for train dataset")
test.process(
    model_name_or_path=model_path_train_last,
    dataset_name_or_path=dataset_path_train_sft,
    split='train',
    max_new_tokens=test_max_new_tokens,
)