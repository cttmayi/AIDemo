
from modelscope.trainers import build_trainer

from modelscope.msdatasets import MsDataset

# train_dataset = MsDataset.load('afqmc_small', split='train')
# eval_dataset = MsDataset.load('afqmc_small', split='validation')

# 载入训练数据，数据格式类似于{'sentence1': 'some content here', 'sentence2': 'other content here', 'label': 0}
train_dataset = MsDataset.load('clue',  subset_name='afqmc', split='train')
eval_dataset = MsDataset.load('clue',  subset_name='afqmc', split='validation')


# 指定工作目录
tmp_dir = "_tmp"

# 指定文本分类模型
model_id = 'damo/nlp_structbert_sentence-similarity_chinese-tiny'
model_id = 'damo/nlp_structbert_backbone_base_std'

def cfg_modify_fn(cfg):
    cfg.preprocessor.type='sen-sim-tokenizer'
    cfg.preprocessor.first_sequence = 'sentence1'
    cfg.preprocessor.second_sequence = 'sentence2'
    cfg.preprocessor.label = 'label'
    cfg.preprocessor.label2id = {'0': 0, '1': 1}
    cfg.model.num_labels = 2
    cfg.task = 'text-classification'
    cfg.pipeline = {'type': 'text-classification'}
    cfg.train.max_epochs = 5
    cfg.train.work_dir = '/tmp'
    cfg.train.dataloader.batch_size_per_gpu = 32
    cfg.evaluation.dataloader.batch_size_per_gpu = 32
    cfg.train.dataloader.workers_per_gpu = 0
    cfg.evaluation.dataloader.workers_per_gpu = 0
    cfg.train.optimizer.lr = 2e-5
    cfg.train.lr_scheduler.total_iters = int(len(train_dataset) / cfg.train.dataloader.batch_size_per_gpu) * cfg.train.max_epochs
    cfg.evaluation.metrics = 'seq-cls-metric'
    # 注意这里需要返回修改后的cfg
    return cfg


# 配置参数
kwargs = dict(
    model=model_id,
    train_dataset=train_dataset,
    # eval_dataset=eval_dataset,
    cfg_modify_fn=cfg_modify_fn,
    # work_dir=tmp_dir,
)


trainer = build_trainer(default_args=kwargs)

trainer.train()