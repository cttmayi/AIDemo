
from utils.data import create_dataset, data_len
from utils.model import create_model, save_model
from trainer import Trainer
import random
import torch
import numpy as np

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
# torch.cuda.manual_seed(0)

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)


device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
model_path = 'AE.pt'


def model_save_callback(models, loss):
    if loss < 1:
        save_model(models, model_path)


if __name__ == '__main__':
    train_loader, test_loader = create_dataset('', 10)
    model = create_model(data_len, data_len, 2, 2000, model_path)
    print(model)

    pretrain_epoch = 10000

    model_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optimizer, T_max=pretrain_epoch)
    model_criterion = torch.nn.MSELoss()

    # Train
    trainer = Trainer(model, train_loader, test_loader, model_optimizer, model_criterion, device, model_save_callback)
    trainer.pretrain(pretrain_epoch)
    rate = trainer.evelate()
    print('error rate: %.2f%%' % (rate * 100))
    save_model(model, 'F.pt')
