import os

from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer


def cfg_modify_fn(cfg):
    cfg.task = 'text-classification'
    cfg.pipeline = {'type': 'text-classification'}
    cfg.preprocessor = {
        'train': {
            # 配置预处理器名字
            'type': 'sen-cls-tokenizer',
            # 配置句子1的key
            'first_sequence': 'sentence',
            # 配置label
            'label': 'label',
            # 配置mode
            'mode': 'train',
        },
        'val': {
            # 配置预处理器名字
            'type': 'sen-cls-tokenizer',
            # 配置句子1的key
            'first_sequence': 'sentence',
            # 配置label
            'label': 'label',
            'mode': 'eval',
        }
    }
    cfg.model['num_labels'] = 15
    cfg['train'] = {
        "work_dir": "/tmp",
        "max_epochs": 10,
        "dataloader": {
            # batch_size
            "batch_size_per_gpu": 16,
            "workers_per_gpu": 0
        },
        "optimizer": {
            # optimizer信息
            "type": "SGD",
            "lr": 0.01,
            "options": {
                "grad_clip": {
                    "max_norm": 2.0
                }
            }
        },
        "lr_scheduler": {
            # lr_scheduler信息，注意torch版本是否包含该lr_scheduler
            "type": "StepLR",
            "step_size": 2,
            "options": {
                "warmup": {
                    "type": "LinearWarmup",
                    "warmup_iters": 2
                }
            }
        },
        "hooks": [{
            "type": "CheckpointHook",
            "interval": 1,
            "by_epoch": False,
        }, {
            "type": "EvaluationHook",
            "interval": 1,
            "by_epoch": False,
        }]
    }
    cfg['evaluation'] = {
        "dataloader": {
            # batch_size
            "batch_size_per_gpu": 16,
            "workers_per_gpu": 0,
            "shuffle": False
        },
        "metrics": [{
            "type": "seq-cls-metric",
            "label_name": "labels",
            "logit_name": "logits",
        }]
    }
    return cfg


dataset = MsDataset.load('clue', subset_name='tnews')

kwargs = dict(
    model='damo/nlp_structbert_backbone_base_std',
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    cfg_modify_fn=cfg_modify_fn)

os.environ['LOCAL_RANK'] = '0'
trainer = build_trainer(name='trainer', default_args=kwargs)
trainer.train()