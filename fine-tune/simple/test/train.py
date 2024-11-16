# structbert的backbone，该模型没有有效分类器，因此使用前需要finetune（微调）
model_id = 'damo/nlp_structbert_backbone_base_std'


from modelscope.msdatasets import MsDataset
# 载入训练数据，数据格式类似于{'sentence1': 'some content here', 'sentence2': 'other content here', 'label': 0}
train_dataset = MsDataset.load('clue',  subset_name='afqmc', split='train')
# 载入评估数据
eval_dataset = MsDataset.load('clue',  subset_name='afqmc', split='validation')

for data in train_dataset:
    print(data)




# from modelscope.msdatasets import MsDataset
# 载入训练数据
# train_dataset = MsDataset.load('/path/to/my_train_file.txt')
# 载入评估数据
# eval_dataset = MsDataset.load('/path/to/my_eval_file.txt')


from modelscope.utils.hub import read_config
# 上面的model_id
cfg = read_config(model_id)

print(cfg)


# 这个方法在trainer读取configuration.json后立即执行，先于构造模型、预处理器等组件
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

from modelscope.trainers import build_trainer

# 配置参数
kwargs = dict(
        model=model_id,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        cfg_modify_fn=cfg_modify_fn)
trainer = build_trainer(default_args=kwargs)
trainer.train()