from tensorboardX import SummaryWriter
import datetime

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.datetime.now())

writer = SummaryWriter('./board/' + TIMESTAMP)


from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl
from transformers.training_args import TrainingArguments


class BoardCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        print('logs:', logs)
        # print('state:', state)
        # print('control:', control)
        
        # print('args:', args)
        if 'loss' in logs and 'epoch' in logs:
            writer.add_scalar('loss', logs['loss'], logs['epoch'])
        if 'learning_rate' in logs and 'epoch' in logs:
            writer.add_scalar('learning_rate', logs['learning_rate'], logs['epoch'])
        #if 'grad_norm' in logs and 'epoch' in logs:
        #    writer.add_scalar('grad_norm', logs['grad_norm'], logs['epoch'])
    

class SaveCallback(TrainerCallback):
    def __init__(self, threshold=0.1):
        super().__init__()
        self.threshold = threshold

        self.best_loss = float('inf')

    def on_log(self, args, state, control, logs=None, **kwargs):
        loss = logs.pop("loss", None)
        if loss is not None:
            if loss < self.best_loss - self.threshold:
                self.best_loss = loss
                control.should_save = True

class NormalCallback(TrainerCallback):
    def __init__(self):
        super().__init__()


    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.epoch % 1 == 0:
            control.should_log = True

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        print('metrics:', metrics)
        pass


    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs, **kwargs):
        loss = logs.pop("loss", None)
        learning_rate = logs.pop("learning_rate", None)
        grad_norm = logs.pop("grad_norm", None)
        epoch = logs.pop("epoch", None)

        # print('loss:', loss)

        if loss is not None and epoch is not None:
            writer.add_scalar('loss', loss, epoch)
        if learning_rate is not None and epoch is not None:
            writer.add_scalar('learning_rate', learning_rate, epoch)
