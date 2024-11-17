import torch

import cfg
import libs.utils as utils
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot

dataset_dir = [cfg.local_dataset_path_test]


if __name__ == '__main__':
    # model, tokenizer, template = utils.init_model(cfg.output_model_path, cfg.template_type)
    model, tokenizer, template = utils.create_model(cfg.local_model_path, cfg.template_type)
    dataset = utils.init_dataset([cfg.local_dataset_path_test], template)

    batch = dataset[0]
    input_ids = batch['input_ids']
    device = next(model.parameters()).device
    input_ids = torch.tensor(input_ids, device=device)[None]
    model_output = model(input_ids)
    

    if False:
        writer = SummaryWriter("./logs/runs")
        writer.add_text('model', str(model))
        writer.add_graph(model, input_ids)
        writer.close()

    g = make_dot(model_output.logits, params=dict(model.named_parameters()))
    g.view()
