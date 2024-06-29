from src import dataset, model, training, test, user
from src import default
import os

from src.utils.callback import BestSaveCallback # EarlyStoppingCallback

from transformers.trainer_utils import (
    EvaluationStrategy,
    FSDPOption,
    HubStrategy,
    IntervalStrategy,
    SchedulerType,
)

NO = 0
YES = 1
FORCE = 2
CONTINUE = 3

CURR_PATH = os.path.dirname(os.path.abspath(__file__))

# CACHE = os.path.join(CURR_PATH, "local_cache")
CACHE = "cache"

MODEL_CACHE = None # os.path.join(CACHE, "model")

PROJECT_CACHE = os.path.join(CACHE, "project")

for path in [CACHE, MODEL_CACHE, PROJECT_CACHE]:
    if path and not os.path.exists(path):
        os.makedirs(path)

def workflow(
    project_name = None,
    # MODEL =================
    model_name_or_path_base = None,
    model_use_8bit_quantization = False,
    model_use_4bit_quantization = False,
    model_device = None,

    # TRAIN PT ===========
    dataset_name_or_path_pt = None,
    dataset_template_pt = None,
    dataset_test_data_size_pt = 0,

    training_use_peft_lora_pt = False,
    training_max_length_pt = None,
    training_args_pt: default.TrainArguments = None,

    # TRAIN SFT ==========
    dataset_name_or_path_sft = None,
    dataset_template_sft = None,
    dataset_test_data_size_sft = 0.1,

    training_repeat_sft = 1,
    training_use_best_sft = False,
    training_use_peft_lora_sft = False,
    training_max_length_sft = None,
    training_args_sft: default.TrainArguments = None,

    # TRAIN Config ===========
    train_use_early_stopping = False,


    # LORA Config ============
    train_lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    train_lora_r = 4,
    train_lora_alpha = 32,
    train_lora_dropout = 0.1,


    # TEST Config ============
    dataset_name_or_path_test = None,
    dataset_template_test = None,

    test_max_new_tokens = 16,

    # Workflow Config: NO; YES; FORCE
    is_dataset_pt = NO,
    is_dataset_sft = NO,
    is_dataset_test = NO,
    is_finetune_pt = NO,
    is_finetune_sft = NO,
    is_test_dataset_test = NO,
    is_test_dataset_train = NO,
    is_test_user = NO,

):
    # config
    if project_name is None:
        raise Exception("project_name is None")

    if model_device is None or model_device == 'auto':
        model_device = default.default_device

    model_name = model_name_or_path_base.split("/")[-1]
    model_path_train_base = os.path.join(PROJECT_CACHE, project_name, 'model', model_name)
    dataset_path_train_base = os.path.join(PROJECT_CACHE, project_name, "dataset")

    dataset_path_train_pt = dataset_path_train_base + "/pt"
    dataset_path_train_sft = dataset_path_train_base + "/sft"
    dataset_path_train_test = dataset_path_train_base + "/test"
    model_path_train_pt = model_path_train_base + "-pt"
    model_path_train_sft = model_path_train_base + "-sft"

    training_use_peft_lora_sft = training_use_peft_lora_sft or training_use_peft_lora_pt
    # =====================================
    model_path_train_last = model_name_or_path_base
    dataset_path_test_last = None

    if model_path_train_base is not None:
        print("### model process for local")
        if not os.path.exists(model_path_train_base):
            model.process(model_path_train_last, model_path_train_base, MODEL_CACHE)
        model_path_train_last = model_path_train_base


    if is_dataset_pt != NO and dataset_name_or_path_pt is not None:
        print("### dataset process for pt")
        if is_dataset_pt == FORCE or not os.path.exists(dataset_path_train_pt):
            dataset.process(dataset_name_or_path_pt, dataset_path_train_pt, model_path_train_last, dataset_template_pt, dataset_test_data_size_pt)

    if is_dataset_sft != NO and dataset_name_or_path_sft is not None:
        print("### dataset process for sft")
        if is_dataset_sft == FORCE or not os.path.exists(dataset_path_train_sft):
            dataset.process(dataset_name_or_path_sft, dataset_path_train_sft, model_path_train_last, dataset_template_sft, dataset_test_data_size_sft)
        dataset_path_test_last = dataset_path_train_sft


    if is_dataset_test != NO and dataset_name_or_path_test is not None:
        print("### dataset process for test")
        if is_dataset_test == FORCE or not os.path.exists(dataset_path_train_test):
            dataset.process(dataset_name_or_path_test, dataset_path_train_test, model_path_train_last, dataset_template_test, 1)
        dataset_path_test_last = dataset_path_train_test

    if is_finetune_pt != NO:
        print("### train process for pt")
        if is_finetune_pt >= FORCE or not os.path.exists(model_path_train_pt):
            if is_finetune_pt == CONTINUE and os.path.exists(model_path_train_pt):
                model_path_train_last = model_path_train_pt
            basic_args = default.BasicArguments(
                dataset_name_or_path=dataset_path_train_pt,
                model_name_or_path=model_path_train_last,
                use_peft_lora=training_use_peft_lora_pt,
                max_seq_length=training_max_length_pt,
                use_8bit_quantization=model_use_8bit_quantization,
                use_4bit_quantization=model_use_4bit_quantization,

                lora_target_modules=train_lora_target_modules,
                lora_r=train_lora_r,
                lora_alpha=train_lora_alpha,
                lora_dropout=train_lora_dropout,

                model_output_dir=model_path_train_pt,
            )

            if dataset_test_data_size_pt > 0:
                training_args_pt.evaluation_strategy="epoch"
            training_args_pt.per_device_eval_batch_size=training_args_pt.per_device_train_batch_size
            training_args_pt.use_cpu=(model_device == "cpu")

            training.process(basic_args, training_args_pt)
        model_path_train_last = model_path_train_pt

    # for _ in range(training_repeat_sft):

    if is_finetune_sft != NO:
        print("### train process for sft")
        if is_finetune_sft >= FORCE or not os.path.exists(model_path_train_sft):
            if is_finetune_sft == CONTINUE and os.path.exists(model_path_train_sft):
                model_path_train_last = model_path_train_sft
            basic_args = default.BasicArguments(
                dataset_name_or_path=dataset_path_train_sft,
                model_name_or_path=model_path_train_last,
                use_peft_lora=training_use_peft_lora_sft,
                max_seq_length=training_max_length_sft,
                use_8bit_quantization=model_use_8bit_quantization,
                use_4bit_quantization=model_use_4bit_quantization,

                lora_target_modules=train_lora_target_modules,
                lora_r=train_lora_r,
                lora_alpha=train_lora_alpha,
                lora_dropout=train_lora_dropout,
                model_output_dir=model_path_train_sft,
            )

            if dataset_test_data_size_sft > 0:
                training_args_sft.evaluation_strategy = IntervalStrategy.EPOCH
            training_args_sft.per_device_eval_batch_size=training_args_sft.per_device_train_batch_size
            training_args_sft.use_cpu=(model_device == "cpu")


            if training_use_best_sft:
                # basic_args.callbacks = [BestSaveCallback()]
                training_args_sft.save_total_limit = 1
                training_args_sft.save_strategy = IntervalStrategy.EPOCH
                training_args_sft.load_best_model_at_end = True
                training_args_sft.metric_for_best_model = 'eval_loss'

            training.process(basic_args, training_args_sft)
        model_path_train_last = model_path_train_sft


    # test process
    if is_test_dataset_test != NO:
        print("test process for test dataset")
        test.process(
            model_name_or_path=model_path_train_last,
            dataset_name_or_path=dataset_path_test_last,
            split='test',
            max_new_tokens=test_max_new_tokens,
            device=model_device,
        )

    if is_test_dataset_train:
        print("test process for train dataset")
        test.process(
            model_name_or_path=model_path_train_last,
            dataset_name_or_path=dataset_path_test_last,
            split='train',
            max_new_tokens=test_max_new_tokens,
            device=model_device,
        )
    
    if is_test_user:
        print("test process for user")
        user.process(
            model_name_or_path=model_path_train_last,
            max_new_tokens=test_max_new_tokens,
            device=model_device,
        )