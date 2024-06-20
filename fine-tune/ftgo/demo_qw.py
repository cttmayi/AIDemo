import os
from workflow import workflow, NO, YES, FORCE, CONTINUE, CURR_PATH
from src import default
from templates.example import SFT, PT
# =====================================


workflow(
    project_name="demo",
    # MODEL =================
    model_name_or_path_base = "Qwen/Qwen1.5-0.5B",
    model_device='mps',

    # TRAIN Config ===========
    train_use_peft_lora = True,
    train_max_length = 512,

    # LORA Config =============
    train_lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    train_lora_r = 4,
    train_lora_alpha = 32,
    train_lora_dropout = 0.1,


    # FINETUNE PT ===========
    dataset_name_or_path_pt = os.path.join(CURR_PATH, "data/example"),
    dataset_template_pt = PT(512),
    dataset_test_data_size_pt = 0,

    training_args_pt = default.TrainArguments(
        num_train_epochs=1,
        per_device_train_batch_size=4,
    ),

    # FINETUNE SFT ==========
    dataset_name_or_path_sft = os.path.join(CURR_PATH, "data/example.jsonl"),
    dataset_template_sft = SFT(16, 8),
    dataset_test_data_size_sft = 0.1,

    training_args_sft = default.TrainArguments(
        num_train_epochs=10,
        per_device_train_batch_size=4,
    ),

    # TEST Config ============
    dataset_name_or_path_test = os.path.join(CURR_PATH, "data/example.jsonl"),
    dataset_template_test = SFT(512),

    test_max_new_tokens = 16,

    # Workflow Config: NO; YES; FORCE; CONTINUE
    is_dataset_pt = FORCE,
    is_dataset_sft = FORCE,
    is_dataset_test = FORCE,
    is_finetune_pt = NO,
    is_finetune_sft = NO,
    is_test_dataset_train=NO,
    is_test_dataset_test=NO
)