import os
from workflow import workflow, NO, YES, FORCE

workflow(
    # MODEL =================
    model_name_or_path_base = "facebook/opt-350m",
    model_path_train = "model/opt-350m",

    # FINETUNE PF ===========
    dataset_name_or_path_pt_base = 'byroneverson/shell-cmd-instruct',
    dataset_path_train_pt = "data/shell-cmd-instruct/pt",
    dataset_template_pt = 'shell_pt',
    dataset_test_data_size_pt = 0,
    model_path_train_pt = "model/opt-350m-pt",
    num_train_epochs_pt = 2,

    # FINETUNE SFT ==========
    dataset_name_or_path_sft_base = 'byroneverson/shell-cmd-instruct',
    dataset_path_train_sft = "data/shell-cmd-instruct/sft",
    dataset_template_sft = 'shell_sft',
    dataset_test_data_size_sft = 0.1,
    model_path_train_sft = "model/opt-350m-sft",
    num_train_epochs_sft = 5,

    # TRAIN Config ===========
    train_use_peft_lora = False,
    train_batch_size = 4,
    train_max_length = 512,

    # TEST Config ============
    test_max_new_tokens = 32,

    # Workflow Config: NO; YES; FORCE
    is_dataset_pt = NO,
    is_dataset_sft = YES,
    is_finetune_pt = NO,
    is_finetune_sft = YES,
)

