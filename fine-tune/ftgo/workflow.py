from src import dataset, model, training, test
from src import default
import os

NO = 0
YES = 1
FORCE = 2
CONTINUE = 3

CURR_PATH = os.path.dirname(os.path.abspath(__file__))


def workflow(
    # MODEL =================
    model_name_or_path_base,
    model_path_train_base = None,
    model_use_8bit_quantization = False,
    model_device = None,

    # DATASET ===============
    dataset_path_train_base = None,

    # TRAIN PT ===========
    dataset_name_or_path_pt_base = None,
    dataset_template_pt = None,
    dataset_test_data_size_pt = 0,
    train_num_train_epochs_pt = 2,

    # TRAIN SFT ==========
    dataset_name_or_path_sft_base = None,
    dataset_template_sft = None,
    dataset_test_data_size_sft = 0.1,
    train_num_train_epochs_sft = 10,

    # TRAIN Config ===========
    train_use_peft_lora = False,
    train_gradient_checkpointing = False,
    train_batch_size = 4,
    train_max_length = 512,
    # train_use_early_stopping = True,

    # LORA Config ============
    train_lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    train_lora_r = 4,
    train_lora_alpha = 32,
    train_lora_dropout = 0.1,


    # TEST Config ============
    test_max_new_tokens = 16,

    # Workflow Config: NO; YES; FORCE
    is_dataset_pt = NO,
    is_dataset_sft = NO,
    is_finetune_pt = NO,
    is_finetune_sft = NO,
    is_test_dataset_test = NO,
    is_test_dataset_train = NO,

):
    # config
    dataset_path_train_pt = dataset_path_train_base + "/pt"
    dataset_path_train_sft = dataset_path_train_base + "/sft"
    model_path_train_pt = model_path_train_base + "-pt"
    model_path_train_sft = model_path_train_base + "-sft"

    # =====================================
    model_path_train_last = model_name_or_path_base

    if model_path_train_base is not None:
        print("model process for local")
        if not os.path.exists(model_path_train_base):
            model.process(model_path_train_last, model_path_train_base)
        model_path_train_last = model_path_train_base

    if is_dataset_pt != NO:
        print("dataset process for pt")
        if is_dataset_pt == FORCE or not os.path.exists(dataset_path_train_pt):
            dataset.process(dataset_name_or_path_pt_base, dataset_path_train_pt, model_path_train_last, dataset_template_pt, dataset_test_data_size_pt)

    if is_dataset_sft != NO:
        print("dataset process for sft")
        if is_dataset_sft == FORCE or not os.path.exists(dataset_path_train_sft):
            dataset.process(dataset_name_or_path_sft_base, dataset_path_train_sft, model_path_train_last, dataset_template_sft, dataset_test_data_size_sft)

    if is_finetune_pt != NO:
        print("train process for pt")
        if is_finetune_pt >= FORCE or not os.path.exists(model_path_train_pt):
            if is_finetune_pt == CONTINUE and os.path.exists(model_path_train_pt):
                model_path_train_last = model_path_train_pt
            basic_args = default.BasicArguments(
                dataset_name_or_path=dataset_path_train_pt,
                model_name_or_path=model_path_train_last,
                use_peft_lora=train_use_peft_lora,
                max_seq_length=train_max_length,
                use_8bit_quantization=model_use_8bit_quantization,
                use_cpu=(model_device == "cpu"),

                lora_target_modules=train_lora_target_modules,
                lora_r=train_lora_r,
                lora_alpha=train_lora_alpha,
                lora_dropout=train_lora_dropout,
            )
            training_args = default.TrainArguments(
                num_train_epochs = train_num_train_epochs_pt,
                model_output_dir=model_path_train_pt,
                per_device_eval_batch_size=train_batch_size,
                per_device_train_batch_size=train_batch_size,
                gradient_checkpointing = train_gradient_checkpointing,
                use_cpu=(model_device == "cpu"),
            )
            training.process(basic_args, training_args)
        model_path_train_last = model_path_train_pt

    if is_finetune_sft != NO:
        print("train process for sft")
        if is_finetune_sft >= FORCE or not os.path.exists(model_path_train_sft):
            if is_finetune_sft == CONTINUE and os.path.exists(model_path_train_sft):
                model_path_train_last = model_path_train_sft
            basic_args = default.BasicArguments(
                dataset_name_or_path=dataset_path_train_sft,
                model_name_or_path=model_path_train_last,
                use_peft_lora=train_use_peft_lora,
                max_seq_length=train_max_length,
                use_8bit_quantization=model_use_8bit_quantization,

                lora_target_modules=train_lora_target_modules,
                lora_r=train_lora_r,
                lora_alpha=train_lora_alpha,
                lora_dropout=train_lora_dropout,
            )
            training_args = default.TrainArguments(
                num_train_epochs=train_num_train_epochs_sft,
                model_output_dir=model_path_train_sft,
                evaluation_strategy="epoch",
                #save_strategy="epoch",
                #save_total_limit=2,
                per_device_eval_batch_size=train_batch_size,
                per_device_train_batch_size=train_batch_size,
                gradient_checkpointing = train_gradient_checkpointing,
                #load_best_model_at_end = True,
            )
            training.process(basic_args, training_args)
        model_path_train_last = model_path_train_sft

    # if is_finetune_pt != NO or is_finetune_sft != NO:
    if is_test_dataset_test != NO:
        print("test process for test dataset")
        test.process(
            model_name_or_path=model_path_train_last,
            dataset_name_or_path=dataset_path_train_sft,
            split='test',
            max_new_tokens=test_max_new_tokens,
            device=model_device,
        )
    if is_test_dataset_train:
        print("test process for train dataset")
        test.process(
            model_name_or_path=model_path_train_last,
            dataset_name_or_path=dataset_path_train_sft,
            split='train',
            max_new_tokens=test_max_new_tokens,
            device=model_device,
        )