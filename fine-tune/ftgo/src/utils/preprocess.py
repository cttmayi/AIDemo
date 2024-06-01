from functools import partial
from itertools import chain
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Tuple

#from ..extras.constants import IGNORE_INDEX
# from ..extras.logging import get_logger
#from .utils import Role



from transformers import Seq2SeqTrainingArguments
from transformers.tokenization_utils import PreTrainedTokenizer

# from ..hparams import DataArguments
# from .template import Template


IGNORE_INDEX = -100

class Preprocess:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer


    def preprocess_sft_pr(self, examples: Dict[str, List[Any]],
        #tokenizer: "PreTrainedTokenizer",
        #template: "Template",
        #data_args: "DataArguments",
    ) -> Dict[str, List[List[int]]]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

        for i in range(len(examples["prompt"])):
            #if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) != 1:
            #    continue
            
            source_ids = self.tokenizer(examples["prompt"][i],
                #add_special_tokens=add_special_tokens,
                truncation=True,
                padding=False,
                #max_length=max_seq_length,
                return_overflowing_tokens=False,
                return_length=False,)
            target_ids = self.tokenizer(examples["response"][i],
                #add_special_tokens=add_special_tokens,
                truncation=True,
                padding=False,
                #max_length=max_seq_length,
                return_overflowing_tokens=False,
                return_length=False,)
            #messages = examples["prompt"][i] + examples["response"][i]
            input_ids, labels = [], []


            source_mask = [IGNORE_INDEX] * len(source_ids)

            input_ids += source_ids + target_ids
            labels += source_mask + target_ids

            #if self.template.efficient_eos:
            input_ids += [self.tokenizer.eos_token_id]
            labels += [self.tokenizer.eos_token_id]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)

        return model_inputs