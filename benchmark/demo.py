import torch
import json
from typing import Dict, List
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, load_dataset
from tqdm import tqdm
import numpy as np

# ====================
# 配置系统（可保存为 YAML/JSON）
# ====================
@dataclass
class ModelConfig:
    name: str
    hf_path: str
    batch_size: int
    max_length: int
    gen_kwargs: dict

@dataclass
class TaskConfig:
    name: str
    dataset_path: str
    dataset_name: str
    dataset_split: str
    input_template: str
    output_processor: str
    metrics: List[str]

BENCHMARK_CONFIG = {
    "models": [
        ModelConfig(
            name="GPT-2 Medium",
            hf_path="gpt2-medium",
            batch_size=8,
            max_length=256,
            gen_kwargs={"max_new_tokens": 50, "do_sample": False}
        ),
        # ModelConfig(
        #     name="Llama-2-0.5b",
        #     hf_path="meta-llama/Llama-2-0.5b-hf",
        #     batch_size=4,
        #     max_length=256,
        #     gen_kwargs={"max_new_tokens": 100, "temperature": 0.7}
        # )
    ],
    "tasks": [
        TaskConfig(
            name="TruthfulQA",
            dataset_path="truthful_qa",
            dataset_name="generation",
            dataset_split="validation",
            input_template="Q: {question}\nA:",
            output_processor="split_after_colon",
            metrics=["accuracy", "bleu"]
        ),
        # TaskConfig(
        #     name="CodeGen",
        #     dataset_path="openai_humaneval",
        #     input_template="Complete this code: {prompt}",
        #     output_processor="extract_code_block",
        #     metrics=["pass@1"]
        # )
    ]
}

# ====================
# 核心系统组件
# ====================
class ModelWrapper:
    """模型加载和推理的抽象层"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(config.hf_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        self.model = AutoModelForCausalLM.from_pretrained(config.hf_path).to(self.device)
        self.model.eval()
    
    def generate(self, prompts: List[str]) -> List[str]:
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **self.config.gen_kwargs
            )
        
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

class TaskProcessor:
    """任务处理抽象层"""
    OUTPUT_PROCESSORS = {
        "split_after_colon": lambda x: x.split("A:")[-1].strip(),
        "extract_code_block": lambda x: x.split("```")[1][len("python\n"):] if "```" in x else x
    }
    
    def __init__(self, config: TaskConfig):
        self.config = config
        self.dataset = load_dataset(config.dataset_path, config.dataset_name)[config.dataset_split]
        print(self.dataset)
        self.output_processor = self.OUTPUT_PROCESSORS[config.output_processor]
    
    def preprocess_input(self, example: dict) -> str:
        """根据模板构造输入"""
        return self.config.input_template.format(**example)
    
    def postprocess_output(self, raw_output: str) -> str:
        """提取模型输出中的有效部分"""
        return self.output_processor(raw_output)

class BenchmarkEvaluator:
    """评估指标计算中心"""
    @staticmethod
    def evaluate(task_name: str, predictions: List[str], references: List[str], metrics: List[str]) -> dict:
        results = {}
        for metric in metrics:
            if metric == "accuracy":
                results[metric] = np.mean([p == r for p, r in zip(predictions, references)])
            elif metric == "bleu":
                results[metric] = BenchmarkEvaluator._calculate_bleu(predictions, references)
            elif metric == "pass@1":
                results[metric] = BenchmarkEvaluator._calculate_code_pass_rate(task_name, predictions)
        return results
    
    @staticmethod
    def _calculate_bleu(predictions, references):
        from sacrebleu import corpus_bleu
        return corpus_bleu(predictions, [references]).score
    
    @staticmethod
    def _calculate_code_pass_rate(task_name, predictions):
        # 这里需要实际执行代码的测试（示例伪代码）
        return 0.0  # 实际应实现代码执行和单元测试验证

# ====================
# 运行流程控制
# ====================
class BenchmarkRunner:
    def __init__(self, config: dict):
        self.config = config
        self.results = []
    
    def run(self):
        for model_config in self.config["models"]:
            model = ModelWrapper(model_config)
            
            for task_config in self.config["tasks"]:
                processor = TaskProcessor(task_config)
                inputs_, predictions_, references_ = [], [], []
                
                # for i in tqdm(range(0, len(processor.dataset), model_config.batch_size)):
                for i in tqdm(range(0, 10, model_config.batch_size)):
                    batch = processor.dataset.select(range(i, min(i+model_config.batch_size, len(processor.dataset))))
                    
                    # 处理输入输出
                    inputs = [processor.preprocess_input(ex) for ex in batch]
                    raw_outputs = model.generate(inputs)
                    processed_outputs = [processor.postprocess_output(o) for o in raw_outputs]

                    inputs_.extend(inputs)
                    predictions_.extend(processed_outputs)
                    # references.extend(batch["answer" if "answer" in batch.column_names else "canonical_solution"])
                    references_.extend(batch["best_answer" if "best_answer" in batch.column_names else "canonical_solution"])
                
                    # print("inputs:", inputs[0])
                    # print("processed_outputs:", processed_outputs[0])
                    # print("references:", batch['best_answer'][0])

                # 计算指标
                metrics = BenchmarkEvaluator.evaluate(
                    task_config.name,
                    predictions_,
                    references_,
                    task_config.metrics
                )
                
                # 记录结果
                self.results.append({
                    "model": model_config.name,
                    "task": task_config.name,
                    "metrics": metrics,
                    "samples": list(zip(inputs_, predictions_, references_))[:3]
                })
    
    def save_report(self, path: str):
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)

# ====================
# 执行入口
# ====================
if __name__ == "__main__":
    runner = BenchmarkRunner(BENCHMARK_CONFIG)
    runner.run()
    runner.save_report("benchmark_report.json")
    print("Benchmark completed. Report saved to benchmark_report.json")