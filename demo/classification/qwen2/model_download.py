
import cfg
from modelscope import snapshot_download

model_type = cfg.model_name
model_dir = cfg.local_model_path

model_dir = snapshot_download(model_type, local_dir=model_dir)