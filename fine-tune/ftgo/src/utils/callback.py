from tensorboardX import SummaryWriter
import datetime

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.datetime.now())

writer = SummaryWriter('./board/' + TIMESTAMP)


from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl
from transformers.training_args import TrainingArguments


class BoardCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        loss = logs.get("loss", None)
        epoch = state.epoch
        if loss is not None:
            writer.add_scalar('loss', loss, epoch)


    def on_evaluate(self, args, state, control, metrics, **kwargs):
        eval_loss = metrics.get("eval_loss", None)
        epoch = state.epoch
        if eval_loss is not None:
            writer.add_scalar('eval_loss', eval_loss, epoch)
