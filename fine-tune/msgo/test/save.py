
from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.preprocessors import Preprocessor, TokenClassificationTransformersPreprocessor

from modelscope.utils.constant import NLPTasks


name = 'qwen/Qwen2-0.5B'

# 传入模型id或模型目录
model = Model.from_pretrained(name)

model.save_pretrained('_model')

# 传入模型id或模型目录
preprocessor = Preprocessor.from_pretrained(name)
preprocessor = Preprocessor.from_pretrained('damo/nlp_structbert_sentence-similarity_chinese-base', first_sequence='sent1', second_sequence='sent2')
preprocessor.save_pretrained('_model')

#word_segmentation = pipeline('chat', model=model, preprocessor=preprocessor)
#input = '今天天气不错，适合出去游玩'
#print(word_segmentation(input))