
"""Custom evaluation tasks for LightEval."""
import numpy as np

from lighteval.metrics.dynamic_metrics import (
    ExprExtractionConfig,
    IndicesExtractionConfig,
    LatexExtractionConfig,
    multilingual_extractive_match_metric,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language

from lighteval.metrics.sample_preparator import GenerativePreparator, LoglikelihoodPreparator, PerplexityPreparator
from lighteval.metrics.utils.metric_utils import (
    CorpusLevelMetric,
    CorpusLevelMetricGrouping,
    Metric,
    MetricCategory,
    MetricGrouping,
    MetricUseCase,
    SampleLevelMetric,
    SampleLevelMetricGrouping,
)
from lighteval.metrics.utils.metric_utils import MetricCategory, MetricUseCase, SampleLevelMetric

# https://github.com/huggingface/lighteval/blob/main/docs/source/adding-a-new-metric.mdx
def custom_metric(predictions: list[str], formatted_doc: Doc, **kwargs) -> bool:
    response = predictions[0]
    print('C', response, '===', formatted_doc.choices[formatted_doc.gold_index])
    return 1 if response == formatted_doc.choices[formatted_doc.gold_index] else 0


def agg_function(items):
    flat_items = [item for item in items ]
    score = sum(flat_items) / len(flat_items)
    return score

my_custom_metric = SampleLevelMetric(
    metric_name="custom_metric_name",
    higher_is_better=True,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.ACCURACY,
    # category={MetricCategory},
    # use_case={MetricUseCase},
    sample_level_fn=custom_metric,
    corpus_level_fn=agg_function,
)


expr_gold_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    fallback_mode="first_match",
    precision=5,
    gold_extraction_target=(ExprExtractionConfig(),),
    # Match boxed first before trying other regexes
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
    aggregation_function=max,
)


# https://github.com/huggingface/lighteval/blob/main/docs/source/adding-a-custom-task.mdx
def aime_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["problem"],
        choices=[line["answer"]],
        gold_index=0,
    )

# Define tasks
aime24 = LightevalTaskConfig(
    name="aime24",
    suite=["custom"],
    prompt_function=aime_prompt_fn,
    hf_repo="HuggingFaceH4/aime_2024",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    # metric=[expr_gold_metric],
    metric=[my_custom_metric],
    version=1,
)


TASKS_TABLE = [
    aime24,
]

# MODULE LOGIC
if __name__ == "__main__":
    print([t for t in TASKS_TABLE])
    print(len(TASKS_TABLE))
