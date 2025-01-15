import torch
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device, save_model_callback=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model.to(self.device)
        self.save_model_callback = save_model_callback

    def pretrain(self, epochs):
        best_loss = 1e9
        best_stop_count = 0

        epoch_loop = range(epochs)
        for epoch in epoch_loop:
            self.model.train()
            running_loss = 0.0

            for i, data in enumerate(self.train_loader, 0):
                inputs = data[0].to(self.device)
                # labels = data[1].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            running_loss = running_loss / len(self.train_loader)
            if best_loss > running_loss:
                best_loss = running_loss
                best_stop_count = 0
                rate = self.evelate(0.2, False)
                print('B[%4d/%d] loss: %.6f rate: %.2f%%' %(epoch + 1, epochs, running_loss, rate*100))
                if self.save_model_callback is not None:
                    self.save_model_callback(self.model, running_loss)
                if rate < 0.2:
                    break
            else:
                best_stop_count += 1

            if best_stop_count >= 20:
                best_stop_count = 0
                rate = self.evelate(0.2, False)
                print('S[%4d/%d] loss: %.6f rate: %.2f%%' %(epoch + 1, epochs, running_loss / len(self.train_loader), rate*100))
                if rate < 0.2:
                    if self.save_model_callback is not None:
                        self.save_model_callback(self.model, running_loss)
                    break

    def check_output(self, inputs, outputs, labels, _print=False):
        ret = True
        _inputs:torch.Tensor = inputs.round().long()
        _outputs:torch.Tensor = outputs.round().long()
        _compare:torch.Tensor = (inputs - outputs).abs()
        _ref = _compare.argmax(dim=1)
        
        _compare:torch.Tensor = (_inputs - _outputs).abs()
        _sum = _compare.sum(dim=1)

        if labels[0].item() == -1:
            if _sum.item() == 0:
                if _print: print('Correct', _inputs[0].cpu().numpy())
            else:
                if _print: print('Wrong', _inputs[0].cpu().numpy(), '\n\t', _compare[0].cpu().numpy())
                ret = False
        else:
            if _ref.item() == labels[0].item() and _sum.item() > 0:
                if _print: print('Correct', _inputs[0].cpu().numpy(), labels[0].item(), '\n\t', _compare[0].cpu().numpy())
            else:
                if _print: print('Wrong', _inputs[0].cpu().numpy(), labels[0].item(), '\n\t', _compare[0].cpu().numpy(), _ref.item())
                ret = False
        return ret

    def evelate(self, error_rate_break=1, _print=True):
        error = 0
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.val_loader, 0):
                inputs:torch.Tensor = data[0].to(self.device)
                labels:torch.Tensor = data[1].to(self.device)
                outputs:torch.Tensor = self.model(inputs)

                if not self.check_output(inputs, outputs, labels, _print):
                    error += 1
                rate = error/(i + 1)
                if rate >= error_rate_break:
                    break
        return rate
            