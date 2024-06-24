import os
from workflow import workflow, NO, YES, FORCE, CONTINUE, CURR_PATH
from src import default
from templates.example import SFT, PT
# =====================================


workflow(
    project_name="demo",
    # MODEL =================
    model_name_or_path_base = "Qwen/Qwen1.5-0.5B",
    model_device='auto',

    # LORA Config =============
    train_lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    train_lora_r = 4,
    train_lora_alpha = 32,
    train_lora_dropout = 0.1,

    # FINETUNE PT ===========
    dataset_name_or_path_pt = os.path.join(CURR_PATH, "data/example"),
    dataset_template_pt = PT(512),
    dataset_test_data_size_pt = 0,

    training_use_peft_lora_pt = True,
    training_max_length_pt = 512,
    training_args_pt = default.TrainArguments(
        num_train_epochs = 1,
        per_device_train_batch_size=1,
    ),

    # FINETUNE SFT ==========
    dataset_name_or_path_sft = os.path.join(CURR_PATH, "data/example.jsonl"),
    dataset_template_sft = SFT(16, 8),
    dataset_test_data_size_sft = 0.1,

    training_use_best_sft = True,
    training_use_peft_lora_sft = True,
    training_max_length_sft = 512,
    training_args_sft = default.TrainArguments(
        num_train_epochs=2,
        per_device_train_batch_size=4,
    ),

    # TEST Config ============
    dataset_name_or_path_test = os.path.join(CURR_PATH, "data/example.jsonl"),
    dataset_template_test = SFT(512),

    test_max_new_tokens = 16,

    # Workflow Config: NO; YES; FORCE; CONTINUE
    is_dataset_pt = YES,
    is_dataset_sft = YES,
    is_dataset_test = YES,
    is_finetune_pt = YES,
    is_finetune_sft = CONTINUE,
    is_test_dataset_test=NO,
    is_test_dataset_train=NO,
)