import os
from workflow import workflow, NO, YES, FORCE, CONTINUE, CURR_PATH
from src import default
from templates.poetry import SFT
import logging, sys

# =====================================
# 设置日志级别为DEBUG，这样所有级别（DEBUG, INFO, WARNING, ERROR, CRITICAL）的日志都会被记录
# logging.basicConfig(level=logging.INFO)
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


workflow(
    project_name="poetry",
    # MODEL =================
    model_name_or_path_base = "Qwen/Qwen1.5-0.5B",
    model_device='mps',
    model_use_8bit_quantization = False,

    # LORA Config =============
    train_lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    train_lora_r = 4,
    train_lora_alpha = 32,
    train_lora_dropout = 0.1,

    # FINETUNE PT ===========
    #dataset_name_or_path_pt = os.path.join(CURR_PATH, ""),
    #dataset_template_pt = PT(512),
    #dataset_test_data_size_pt = 0,

    training_use_peft_lora_pt = True,
    training_max_length_pt = 512,
    training_args_pt = default.TrainArguments(
        num_train_epochs = 1,
        per_device_train_batch_size=1,
    ),

    # FINETUNE SFT ==========
    dataset_name_or_path_sft = "Iess/chinese_modern_poetry",
    dataset_template_sft = SFT(),
    dataset_test_data_size_sft = 0,

    training_use_best_sft = True,
    training_use_peft_lora_sft = True,
    training_max_length_sft = 512,
    training_args_sft = default.TrainArguments(
        num_train_epochs=1,
        per_device_train_batch_size=4,
    ),

    # TEST Config ============
    #dataset_name_or_path_test = os.path.join(CURR_PATH, "data/example.jsonl"),
    #dataset_template_test = SFT(512),

    test_max_new_tokens = 128,

    # Workflow Config: NO; YES; FORCE; CONTINUE
    is_dataset_pt = NO,
    is_dataset_sft = YES,
    is_dataset_test = NO,
    is_finetune_pt = NO,
    is_finetune_sft = FORCE,
    is_test_dataset_test = NO,
    is_test_dataset_train = NO,
    is_test_user=YES,
)