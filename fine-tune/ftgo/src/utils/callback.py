from tensorboardX import SummaryWriter
import datetime
import numpy as np

from typing import Any, Dict, List, Optional, Union
from transformers.trainer_utils import (
    EvaluationStrategy,
    FSDPOption,
    HubStrategy,
    IntervalStrategy,
    SchedulerType,
)



from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl
from transformers.training_args import TrainingArguments


class BoardCallback(TrainerCallback):
    def __init__(self):
        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.datetime.now())
        self.writer = SummaryWriter('./board/' + TIMESTAMP)

    def on_log(self, args, state, control, logs=None, **kwargs):
        loss = logs.get("loss", None)
        epoch = state.epoch
        if loss is not None:
            self.writer.add_scalar('loss', loss, epoch)


    def on_evaluate(self, args, state, control, metrics, **kwargs):
        eval_loss = metrics.get("eval_loss", None)
        epoch = state.epoch
        if eval_loss is not None:
            self.writer.add_scalar('eval_loss', eval_loss, epoch)


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, early_stopping_patience: int = 1, early_stopping_threshold: Optional[float] = 0.0):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        # early_stopping_patience_counter denotes the number of times validation metrics failed to improve.
        self.early_stopping_patience_counter = 0

    def check_metric_value(self, args, state, control, metric_value):
        # best_metric is set by code for load_best_model
        operator = np.greater if args.greater_is_better else np.less
        if state.best_metric is None or (
            operator(metric_value, state.best_metric)
            and abs(metric_value - state.best_metric) > self.early_stopping_threshold
        ):
            self.early_stopping_patience_counter = 0
        else:
            self.early_stopping_patience_counter += 1

    def on_train_begin(self, args, state, control, **kwargs):
        pass


    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_to_check = 'loss'# args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics.get(metric_to_check)

        self.check_metric_value(args, state, control, metric_value)
        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True

class BestSaveCallback(TrainerCallback):
    def __init__(self):
        pass

    def check_metric_value(self, args, state, control, metric_value):
        # best_metric is set by code for load_best_model
        operator = np.greater if args.greater_is_better else np.less
        if state.best_metric is None or operator(metric_value, state.best_metric):
            return True

        return False

    def on_train_begin(self, args, state, control, **kwargs):
        pass


    def on_evaluate(self, args, state, control:TrainerControl, metrics, **kwargs):
        metric_to_check = 'eval_loss'
        metric_value = metrics.get(metric_to_check)
        # print('metric_value', metric_value)
        if self.check_metric_value(args, state, control, metric_value):
            # print('save best model')
            control.should_save = True