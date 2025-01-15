
import utils.data as data
import utils.model as model
import trainer as trainer
import random
import torch

random.seed(0)
torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_path = 'AE.pt'

def save_model(models, loss):
    if loss < 0.1:
        model.save_model(models, model_path)

def create_model():
    models = model.create_model(21, 21, 2000, 200, model_path)
    return models

if __name__ == '__main__':

    save_model = True
    # Load data
    train_loader, test_loader = data.create_dataset('', 10)
    # Create model
    models = create_model()

    optimizer = torch.optim.AdamW(models.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()

    # Train
    trainer = trainer.Trainer(models, train_loader, test_loader, optimizer, criterion, device, save_model)
    trainer.pretrain(100000)
    rate = trainer.evelate()
    print('error rate: %.2f%%' % (rate * 100))
    # save_model(models)
