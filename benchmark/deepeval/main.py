from deepeval.benchmarks import MMLU
from deepeval.benchmarks.mmlu.task import MMLUTask

# 定义基准测试任务和 few-shot 数量
benchmark = MMLU(
    tasks=[MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE, MMLUTask.ASTRONOMY],
    n_shots=3
)

# 替换为自己的模型
benchmark.evaluate(model=mistral_7b)
print(benchmark.overall_score)