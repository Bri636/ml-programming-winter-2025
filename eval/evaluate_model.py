""" Evaluation on Human Eval """

from __future__ import annotations
from typing import TypedDict, Dict, List, Any
from transformers import pipeline
from transformers.pipelines import Pipeline
from functools import partial 
from pydantic import Field
from tqdm import tqdm
import timeit
import os, glob
from argparse import ArgumentParser
# submods
from human_eval.data import write_jsonl, read_problems, HUMAN_EVAL
from ml_program.utils import BaseConfig, ConfigLike, batch_data, format_time
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
    model_name_or_path: str = 'unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit'
    task: str = 'text-generation'
    device_map: str = 'auto'
    torch_dtype: str = 'bfloat16'
    max_new_tokens: int = 128
    truncation: bool = True

class EvaluationConfig(BaseConfig): 
    human_eval_path: str = Field(default=HUMAN_EVAL)
    pipeline_config: HFGeneratorConfig = Field(default_factory=HFGeneratorConfig)
    eval_save_dir: str = Field(default='/lus/eagle/projects/FoundEpidem/bhsu/2024_research/ml-program-winter-2025/eval/eval_results')
    num_samples_per_task: int = Field(default=4)
    batch_size: int = Field(default=16)  # Save every 10 samples

# def evaluate(pipeline: Pipeline, 
#              problems: Dict[str, HumanEvalProblem], 
#              eval_config: EvaluationConfig) -> List[HumanEvalSolution]: 
#     """ Evaluation of HFModel """
#     samples = []
#     for idx, (task_id, task_data) in tqdm(enumerate(problems.items())):
#         print(f'Starting Question: {idx + 1}.....')
#         start_time = timeit.default_timer()
#         batched_prompts = [task_data["prompt"]] * eval_config.num_samples_per_task
#         completions = pipeline(batched_prompts, 
#                                max_new_tokens=eval_config.pipeline_config.max_new_tokens, 
#                                truncation=eval_config.pipeline_config.truncation)
#         for completion in completions:
#             samples.append({"task_id": task_id, "completion": completion})
            
#         elapsed_time = timeit.default_timer() - start_time
#         formatted_time = format_time(elapsed_time)
#         print(f"Iteration {idx + 1}: {formatted_time} (HH:MM:SS)\n")

#     return samples

def evaluate(pipeline: Pipeline, 
             problems: Dict[str, HumanEvalProblem], 
             eval_config: EvaluationConfig): 
    """ Evaluation of HFModel with batch saving """
    
    os.makedirs(eval_config.eval_save_dir, exist_ok=True)  # Ensure save directory exists
    batch = []
    file_index = len(glob.glob(os.path.join(eval_config.eval_save_dir, "samples_*.jsonl"))) + 1
    
    for idx, (task_id, task_data) in tqdm(enumerate(problems.items()), total=len(problems)):
        print(f'Starting Question: {idx + 1}...\n')
        start_time = timeit.default_timer()
        
        batched_prompts = [task_data["prompt"]] * eval_config.num_samples_per_task
        completions = pipeline(batched_prompts, 
                               max_new_tokens=eval_config.pipeline_config.max_new_tokens, 
                               truncation=eval_config.pipeline_config.truncation)
        
        for completion in completions:
            batch.append({"task_id": task_id, "completion": completion})
        
        elapsed_time = timeit.default_timer() - start_time
        print(f"Iteration {idx + 1}: {elapsed_time:.6f} seconds\n")
        
        # Save every `batch_size` samples
        if len(batch) >= eval_config.batch_size:
            save_path = os.path.join(eval_config.eval_save_dir, f"samples_{file_index}.jsonl")
            write_jsonl(save_path, batch)
            print(f"Saved {len(batch)} samples to {save_path}")
            batch.clear()
            file_index += 1

    # Save remaining batch if not empty
    if batch:
        save_path = os.path.join(eval_config.eval_save_dir, f"samples_{file_index}.jsonl")
        write_jsonl(save_path, batch)
        print(f"Saved {len(batch)} final samples to {save_path}")

def parse_arguments(): 
    argparser = ArgumentParser()
    argparser.add_argument('--model_name_or_path', type=str, default='unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit')
    argparser.add_argument('--eval_save_dir', type=str, default='/lus/eagle/projects/FoundEpidem/bhsu/2024_research/ml-program-winter-2025/eval/eval_results')
    argparser.add_argument('--merge', type=bool, default=False)
    argparser.add_argument('--merge_output', 
                           type=str, 
                           default='/lus/eagle/projects/FoundEpidem/bhsu/2024_research/ml-program-winter-2025/eval/eval_results/merged.jsonl')
    return argparser.parse_args()


def merge_jsonl_files(input_dir: str, output_file: str):
    """
    Merges multiple JSONL files from a directory into a single JSONL file.
    """
    jsonl_files = sorted(glob.glob(os.path.join(input_dir, "samples_*.jsonl")))

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file in jsonl_files:
            with open(file, 'r', encoding='utf-8') as infile:
                for line in infile:
                    outfile.write(line)

    print(f"Merged {len(jsonl_files)} files into {output_file}")

""" 
Notes: 15 seconds per batch of size 4 & 164 questions ==> (15 * 164) / 60 = 41 minutes for generation of pass@1 only
TODO: save time of jsonl == ??? 
Do batch saving every n steps, then a merge 
Perhaps run with Qwen 2 or Zephyer
"""

def main(): 
    args = parse_arguments()

    if args.merge:
        merge_jsonl_files(args.eval_save_dir, args.merge_output)
        return

    eval_config = EvaluationConfig()
    eval_config.eval_save_dir = args.eval_save_dir
    pipe_config = eval_config.pipeline_config
    pipe_config.model_name_or_path = args.model_name_or_path
    
    problems: Dict[str, HumanEvalProblem] = read_problems()
    pipe = pipeline(model=pipe_config.model_name_or_path, 
                    task='text-generation', 
                    device_map=pipe_config.device_map,
                    torch_dtype=_DTYPES.get(pipe_config.torch_dtype))
    
    evaluate(pipe, problems, eval_config)
    
# TODO: make sure to 
    
if __name__=="__main__": 
    
    main()
    
    # eval_config = EvaluationConfig()
    # pipe_config = eval_config.pipeline_config
    
    # problems: Dict[str, HumanEvalProblem] = read_problems()
    # pipe = pipeline(model=pipe_config.model_name_or_path, 
    #                 task='text-generation', 
    #                 device_map=pipe_config.device_map,
    #                 torch_dtype=_DTYPES.get(pipe_config.torch_dtype))
    
    # samples = evaluate(pipe, problems, eval_config)
    # write_jsonl(eval_config.samples_save_path, samples)