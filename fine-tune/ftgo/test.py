import os
from workflow import workflow, NO, YES, FORCE, CONTINUE, CURR_PATH

# =====================================

workflow(
    # MODEL =================
    model_name_or_path_base = "facebook/opt-350m",
    model_path_train_base = "model/opt-350m",

    # DATASET ===============
    dataset_path_train_base = "data/test",

    # FINETUNE PF ===========
    dataset_name_or_path_pt_base = os.path.join(CURR_PATH, "data/example"),
    #dataset_template_pt = 'example_pt',
    # dataset_test_data_size_pt = 0,
    train_num_train_epochs_pt = 1,

 
    # Workflow Config: NO; YES; FORCE
    is_dataset_pt = FORCE,
    is_dataset_sft = NO,
    is_finetune_pt = NO,
    is_finetune_sft = NO,
    is_test_dataset_train=NO,
    is_test_dataset_test=NO
)