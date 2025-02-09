import os
from typing import List
import numpy as np
from model import ModelBase, ModelOpenAI
from tqdm import tqdm


'''
EM(Exact Match)
定义: EM 表示完全匹配率，即模型生成的答案与标准答案完全一致的比例。
应用场景：常用于问答任务、机器翻译等，要求模型输出与真实答案高度一致的场景。

F1
定义: F1 分数是精确率(Precision)和召回率(Recall)的调和平均值。它衡量模型预测的答案与真实答案的重叠程度，具体计算公式为：
F1 = 2 * (Precision + Recall)/(Precision * Recall)
应用场景：适用于需要综合考虑预测准确性和覆盖范围的任务，如信息检索、文本分类等。

PASS@1
定义: PASS@1 表示模型在第一次尝试时生成的解决方案通过测试的比例。它衡量模型首次生成的代码或答案是否正确。
应用场景：主要用于代码生成任务，如 HumanEval，也适用于其他需要生成准确答案的任务。

BPB(Bits Per Byte)
定义: BPB 通常用于衡量模型的压缩效率或信息熵。它表示每个字节所需的比特数，数值越低，表示模型对数据的压缩效果越好。
应用场景: 在语言模型中，BPB 可用于评估模型对文本数据的编码效率，帮助优化模型的性能。


BLEU(Bilingual Evaluation Understudy): 用于评估机器翻译的质量，通过比较模型生成的翻译与参考翻译的相似度来衡量。
ROUGE（Recall-Oriented Understudy for Gisting Evaluation）：主要用于文本摘要任务，衡量生成摘要与参考摘要的相似度。
ACC（Accuracy）：准确率，表示模型预测正确的比例。
MRR（Mean Reciprocal Rank）：平均倒数排名，衡量模型生成的多个答案中，正确答案的排名倒数的平均值。
PPL（Perplexity）：困惑度，衡量语言模型对文本的预测不确定性。数值越低，表示模型对文本的预测越准确。

'''





class Evaluator:
    def __init__(self, metrics: List[str], llm_generate=None):
        self.metrics = metrics


        if llm_generate is None:
            self.llm = ModelOpenAI(os.getenv('OPENAI_API_KEY'), os.getenv('OPENAI_API_BASE'), 'gpt-3.5-turbo')
            self.llm_generate = self._default_llm_generate
        else:
            self.llm_generate = llm_generate


    def _default_llm_generate(self, prediction: str, reference: str):
        message = f"请判断如下回答是否正确. 如果正确回答, 请回复'1', 否则回复'0'. \n\n 回答：{prediction}, 正确答案：{reference}"

        ret = self.llm.generate([message])[0]
        if ret[0] == "1":
            score = 1.0
        elif ret[0] == "0":
            score = 0.0
        else:
            score = 0.5

        return score


    def evaluate(self, predictions: List[str], references: List[str]) -> dict:
        metrics = self.metrics

        results = {}
        for metric in metrics:
            if metric == 'acc' or metric == "accuracy" or metric == "em" or metric == "exact_match":
                results[metric] = np.mean([p == r for p, r in zip(predictions, references)])
            elif metric == "bleu":
                results[metric] = self._calculate_bleu(predictions, references)
            elif metric == "pass@1":
                results[metric] = self._calculate_code_pass_rate(predictions, references)
            elif metric == "llm":
                results[metric] = self._calculate_llm(predictions, references)
            results[metric] = round(results[metric], 4) * 100
        return results


    def _calculate_bleu(self, predictions, references):
        from sacrebleu import corpus_bleu
        return corpus_bleu(predictions, [references]).score


    def _calculate_code_pass_rate(self, predictions, references):
        # 这里需要实际执行代码的测试（示例伪代码）
        return 0.0  # 实际应实现代码执行和单元测试验证

    def _calculate_llm(self, predictions, references):
        score = 0.0
        for prediction, reference in tqdm(zip(predictions, references), desc='llm evaluation'):
            score += self.llm_generate(prediction, reference)

        return score / len(predictions)
