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
                data = data[0].to(self.device)
                data_err = data[1].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, data)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            running_loss = running_loss / len(self.train_loader)
            if best_loss > running_loss:
                best_loss = running_loss
                best_stop_count = 0
                rate = self.evelate(0.4, True)
                print('B[%4d/%d] loss: %.6f rate: %.2f%%' %(epoch + 1, epochs, running_loss, rate*100))
                if self.save_model_callback is not None:
                    self.save_model_callback(self.model, running_loss)

                if rate < 0.001 and running_loss < 0.001:
                    break
                
            else:
                best_stop_count += 1

            if best_stop_count >= 100:
                best_stop_count = 0
                break

    def check_output(self, inputs, outputs, data_crt, _print=False):
        ret = True
        inputs_long:torch.Tensor = inputs.round().long()
        outputs_long:torch.Tensor = outputs.round().long()
        data_crt_long:torch.Tensor = data_crt.round().long()

        inputs_is_crt = torch.eq(inputs_long, data_crt_long).all().item()
        if not inputs_is_crt:
            i_err_indexs = torch.where(inputs_long != data_crt_long)[1].tolist()
            # print('__', inputs_long[0].cpu().tolist(), data_crt_long[0].cpu().tolist(), i_err_indexs)

        io_compare:torch.Tensor = (inputs - outputs).abs()
        o_err_index = io_compare.argmax(dim=1).item()
        o_err_value = io_compare[0][o_err_index].item()
        if o_err_value < 0.5:
            o_err_index = -1
        
        io_compare_long:torch.Tensor = (inputs_long - outputs_long).abs()

        if inputs_is_crt:
            if o_err_index == -1:
                if _print: print('C ', inputs_long[0].cpu().tolist())
            else:
                if _print: print('W ', inputs_long[0].cpu().numpy(), '__', io_compare_long[0].cpu().tolist())
                ret = False
        else:
            if o_err_index in i_err_indexs:
                if _print: print('C ', inputs_long[0].cpu().tolist(), i_err_indexs, '__', io_compare_long[0].cpu().tolist())
            elif io_compare[0][i_err_indexs[0]].item() > 0.5:
                if _print: print('C_', inputs_long[0].cpu().tolist(), i_err_indexs, '__', io_compare[0].cpu().numpy())
            else:
                if _print: print('W ', inputs_long[0].cpu().tolist(), i_err_indexs, '__', io_compare[0].cpu().numpy(), o_err_index)
                ret = False
        return ret

    def evelate(self, error_rate_break=1, _print=True):
        error = 0
        self.model.eval()
        with torch.no_grad():
            for i, idata in enumerate(self.val_loader, 0):
                data:torch.Tensor = idata[0].to(self.device)
                outputs:torch.Tensor = self.model(data)

                if not self.check_output(data, outputs, data, _print):
                    error += 1

                data_err:torch.Tensor = idata[1].to(self.device)
                outputs:torch.Tensor = self.model(data_err)

                if not self.check_output(data_err, outputs, data, _print):
                    error += 1

                rate = error/((i + 1) * 2)

                if rate >= error_rate_break:
                    break
        return rate
            