# https://blog.csdn.net/m0_46144891/article/details/118710163

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
import re


words = ['它由两部分组成，TF和IDF。TF-IDF是一种加权技术，采用一种统计方法，根据字词在文本中出现的次数和在整个语料中出现的文档频率来计算一个字词在整个语料中的重要程度......']

list_words = []
for line in words:
    # 清洗数据
    text = ''.join(line.split())
    # 实现目标文本中对正则表达式中的模式字符串进行替换
    text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）～-]+", "", text)
    # 利用jieba包自动对处理后的文本进行分词
    test_list = jieba.cut(text, cut_all=False)
    # 得到所有分解后的词
    list_words.append(' '.join(test_list))
print(list_words)


tfidf_vec = TfidfVectorizer(stop_words=None)
"""
相关参数
token_pattern：使用正则表达式进行分词。
max_df/min_df：用于过滤出现太多的无意义词语。
stop_words：list类型，直接过滤指定的停用词。
vocabulary：dict类型，值使用特定的词汇，制定对应关系。
"""
tfidf_matrix = tfidf_vec.fit_transform(list_words)

# 得到语料库所有不重复的词
print(tfidf_vec.get_feature_names_out())

# 得到每个单词对应的id值
print(tfidf_vec.vocabulary_)



# 输出generator
print(tfidf_matrix)
# 得到每个句子对应的向量
print(tfidf_matrix.toarray())
