import torch
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model.to(self.device)

    def pretrain(self, epochs):
        best_loss = 1e9
        best_stop_count = 0

        def evelate():
            ret = 0
            self.model.eval()
            with torch.no_grad():
                for i, data in enumerate(self.val_loader, 0):
                    inputs:torch.Tensor = data[0].to(self.device)
                    labels:torch.Tensor = data[1].to(self.device)
                    outputs:torch.Tensor = self.model(inputs)

                    _inputs:torch.Tensor = inputs.round().long()
                    _outputs:torch.Tensor = outputs.round().long()
                    _compare:torch.Tensor = (inputs - outputs).abs()
                    _ref = _compare.argmax(dim=1)
                    
                    _compare:torch.Tensor = (_inputs - _outputs).abs()
                    _sum = _compare.sum(dim=1)
                    
        
                    if labels[0].item() == -1:
                        if _sum.item() == 0:
                            print('Correct', _inputs[0].cpu().numpy())
                        else:
                            print('Wrong', _inputs[0].cpu().numpy(), _compare[0].cpu().numpy())
                            ret += 1
                    else:
                        if _ref.item() == labels[0].item() and _sum.item() > 0:
                            print('Correct', _inputs[0].cpu().numpy(), _compare[0].cpu().numpy(), labels[0].item())
                        else:
                            print('Wrong', _inputs[0].cpu().numpy(), _compare[0].cpu().numpy(), labels[0].item(), _ref.item())
                            ret += 1
            return ret

        # epoch_loop = tqdm(range(epochs), desc = f'Training {self.model.__class__.__name__}')
        epoch_loop = range(epochs)
        for epoch in epoch_loop:
            self.model.train()
            running_loss = 0.0

            for i, data in enumerate(self.train_loader, 0):
                inputs = data[0].to(self.device)
                labels = data[1].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            # print('[%d, %5d] loss: %.9f' %(epoch + 1, i + 1, running_loss / len(self.train_loader)))

            if best_loss > running_loss:
                best_loss = running_loss
                best_stop_count = 0
                print('[%4d/%d] loss: %.9f' %(epoch + 1, epochs, running_loss / len(self.train_loader)))
            else:
                best_stop_count += 1
                

            if best_stop_count >= 10:
                best_stop_count = 0
                print('Early stop')
                if evelate() == 0:
                    break


            