from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.preprocessors import Preprocessor, TokenClassificationTransformersPreprocessor

from modelscope.utils.constant import NLPTasks


name = 'qwen/Qwen2-0.5B'

# 传入模型id或模型目录
model = Model.from_pretrained(name)

# 传入模型id或模型目录
preprocessor = Preprocessor.from_pretrained(model.model_dir)

word_segmentation = pipeline('chat', model=model, preprocessor=preprocessor)
input = '今天天气不错，适合出去游玩'
print(word_segmentation(input))