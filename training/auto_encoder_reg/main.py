
import utils.data as data
import utils.model as model
import trainer as trainer

import torch

if __name__ == '__main__':
    # Load data
    train_loader, test_loader = data.create_dataset('', 10)
    # Create model
    models = model.create_model(10, 10, 1000, 10, 'AE.pt') 

    optimizer = torch.optim.AdamW(models.parameters(), lr=1e-5)
    criterion = torch.nn.MSELoss()

    # Train
    trainer = trainer.Trainer(models, train_loader, test_loader, optimizer, criterion, 'mps')
    trainer.pretrain(1500)

    model.save_model(models, 'AE.pt')
