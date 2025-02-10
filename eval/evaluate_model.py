""" Evaluation on Human Eval """

from __future__ import annotations
from typing import TypedDict, Dict, List, Any
from transformers import pipeline
from transformers.pipelines import Pipeline
from functools import partial 
from pydantic import Field

# submods
from human_eval.data import write_jsonl, read_problems
from ml_program.utils import BaseConfig, ConfigLike
from ml_program.train import _DTYPES

class HumanEvalProblem(TypedDict): 
    """ One single problem in HumanEval """
    task_id: str
    prompt: str
    entry_point: str
    canonical_solution: str
    test: str
    
class HumanEvalSolution(TypedDict): 
    """ One instance of solution to HumanEval problem """
    task_id: str
    completion: str

class HFGeneratorConfig(BaseConfig): 
    model_name_or_path: str
    task: str = 'text-generation'
    device_map: str = 'auto'
    torch_dtype: str = 'bfloat16'

class EvaluationConfig(BaseConfig): 
    human_eval_path: str
    pipeline_config: HFGeneratorConfig
    
def generate_one_completion(pipeline: Pipeline, prompt: str) -> str: 
    ...
    
if __name__=="__main__": 
    
    pipe = pipeline(model='unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit', 
                    task='text-generation', 
                    device='auto', 
                    torch_dtype=_DTYPES.get('bfloat16'))

    generate_one_completion = partial(generate_one_completion, pipe)
    problems: Dict[str, HumanEvalProblem] = read_problems()
    
    num_samples_per_task = 200
    samples = []
    for _ in range(num_samples_per_task): 
        for task_id in problems: 
            samples.append(dict(task_id=task_id, 
                                completion=generate_one_completion(problems[task_id]["prompt"])))
            
    breakpoint()
    write_jsonl("samples.jsonl", samples)
    
    # num_samples_per_task = 200
    # samples = [
    #     dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
    #     for task_id in problems
    #     for _ in range(num_samples_per_task)
    # ]
    # write_jsonl("samples.jsonl", samples)
    
    breakpoint()