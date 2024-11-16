# https://modelscope.cn/docs/%E6%95%B0%E6%8D%AE%E9%9B%86%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97

from modelscope.msdatasets import MsDataset
# 载入训练数据，数据格式类似于{'sentence1': 'some content here', 'sentence2': 'other content here', 'label': 0}
ds = MsDataset.load('clue',  subset_name='afqmc', split='train')

print(next(iter(ds)))
# 载入评估数据
# eval_dataset = MsDataset.load('clue',  subset_name='afqmc', split='validation')


