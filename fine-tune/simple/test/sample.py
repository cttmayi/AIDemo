from modelscope.msdatasets import MsDataset
from modelscope.pipelines import pipeline

inputs = ['今天天气不错，适合出去游玩', '这本书很好，建议你看看']
dataset = MsDataset.load(inputs, target='sentence')
word_segmentation = pipeline('word-segmentation')
outputs = word_segmentation(dataset)
for o in outputs:
    print(o)

# 输出
# {'output': ['今天', '天气', '不错', '，', '适合', '出去', '游玩']}
# {'output': ['这', '本', '书', '很', '好', '，', '建议', '你', '看看']}


from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.preprocessors import Preprocessor, TokenClassificationTransformersPreprocessor

model = Model.from_pretrained('damo/nlp_structbert_word-segmentation_chinese-base')
tokenizer = Preprocessor.from_pretrained(model.model_dir)
# Or call the constructor directly: 
# tokenizer = TokenClassificationTransformersPreprocessor(model.model_dir)
word_segmentation = pipeline('word-segmentation', model=model, preprocessor=tokenizer)
input = '今天天气不错，适合出去游玩'
print(word_segmentation(input))
# {'output': ['今天', '天气', '不错', '，', '适合', '出去', '游玩']}