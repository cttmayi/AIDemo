import os
from workflow import workflow, NO, YES, FORCE, CONTINUE, CURR_PATH

# =====================================

workflow(
    # MODEL =================
    model_name_or_path_base = "Qwen/Qwen1.5-0.5B",
    model_path_train_base = "model/Qwen1.5-0.5B",
    model_device='cpu',

    # DATASET ===============
    dataset_path_train_base = "data/example",

    # FINETUNE PF ===========
    dataset_name_or_path_pt_base = os.path.join(CURR_PATH, "data/example.jsonl"),
    dataset_template_pt = 'example_pt',
    dataset_test_data_size_pt = 0,
    train_num_train_epochs_pt = 1,

    # FINETUNE SFT ==========
    dataset_name_or_path_sft_base = os.path.join(CURR_PATH, "data/example.jsonl"),
    dataset_template_sft = 'example_sft',
    dataset_test_data_size_sft = 0.1,
    train_num_train_epochs_sft = 2,

    # TRAIN Config ===========
    train_use_peft_lora = True,
    train_batch_size = 4,
    train_max_length = 512,

    # TEST Config ============
    test_max_new_tokens = 16,

    # Workflow Config: NO; YES; FORCE
    is_dataset_pt = YES,
    is_dataset_sft = YES,
    is_finetune_pt = NO,
    is_finetune_sft = NO,
    is_test_dataset_train=YES,
    is_test_dataset_test=NO
)