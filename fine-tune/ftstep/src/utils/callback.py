from tensorboardX import SummaryWriter

writer = SummaryWriter('./board')


from transformers.trainer_callback import TrainerCallback

class PrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        print('logs:', logs)
        print('state:', state)
        print('control:', control)
        
        # print('args:', args)
        if 'loss' in logs and 'epoch' in logs:
            writer.add_scalar('loss', logs['loss'], logs['epoch'])
        if 'learning_rate' in logs and 'epoch' in logs:
            writer.add_scalar('learning_rate', logs['learning_rate'], logs['epoch'])
        if 'grad_norm' in logs and 'epoch' in logs:
            writer.add_scalar('grad_norm', logs['grad_norm'], logs['epoch'])
        #_ = logs.pop("total_flos", None)
        #if state.is_local_process_zero:
        #    print(logs)
    


