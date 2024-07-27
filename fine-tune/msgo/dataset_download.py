import os
import cfg
from modelscope.msdatasets import MsDataset

# ds = MsDataset.load('clue',  subset_name='afqmc', split='train')

ds = MsDataset.load(cfg.dataset_name, split='train')

if isinstance(ds, MsDataset):
    ds = ds.to_hf_dataset()
    

try:
    ds_test = MsDataset.load(cfg.dataset_name, split='validation')
except:  # noqa: E722
    ds_ = ds.train_test_split(test_size=cfg.dataset_split_ratio_test)
    ds = ds_['train']
    ds_test = ds_['test']

print(next(iter(ds)))


ds.to_json(os.path.join(cfg.local_dataset_path_train), force_ascii=False)
ds_test.to_json(os.path.join(cfg.local_dataset_path_test), force_ascii=False)