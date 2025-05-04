import cfg
import libs.utils as utils
from swift import Seq2SeqTrainer, Seq2SeqTrainingArguments

if __name__ == '__main__':
    from swift.llm import sft_main, TrainArguments
    result = sft_main(TrainArguments(
        model=cfg.local_model_path,
        train_type='lora',
        dataset=[
            cfg.local_dataset_path_test,
        ],
        # torch_dtype='bfloat16',
        torch_dtype='float32',
        # ...
    ))