""" Processing dataset for coding datasets """

from __future__ import annotations
from typing import List, Dict, Any
from datasets import Dataset, load_from_disk, load_dataset
from enum import Enum

from ml_program.utils import get_cpu_count

DPOHFBatch = List[Dict[str, List[Dict[str, str]]]]
""" Batch of datast inputs from huggingface as a list """


class HFDatasets:
    CODEFORCE_BIN = 'coseal/CodeUltraFeedback_binarized'
    """Link: https://huggingface.co/datasets/coseal/CodeUltraFeedback_binarized"""


class HFModels:
    QWEN = "Qwen/Qwen2-0.5B-Instruct"

# TODO: pre-process the dataset so that it matches this one right here
# MUST MATCH THIS FORMAT: https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized?row=0

# NOTE: for batched=True, input is a dict of lists; non-batched means list of dicts. it is flipped for performance but is annoying


def process_data(batch: Dict[str, List]) -> DPOHFBatch:
    """ 
    Processes the data to be in the format for CPO and DPO 
    Input format: 
    ========
    Each row ['instruction': str, 'chosen': str, 'rejected': str]

    Output must be format of this dataset: Format of the trl-lib/ultrafeedback_binarized dataset: 
    =====
    Each Row: ['chosen': chosen, 'rejected': rejected, 'score_chosen': ..., 'score_rejected': ...]

    chosen: List[dict{'role': 'user', 'content': ...}, dict{'role': 'assistant', 'content': ...}]
    rejected: List[dict{'role': 'user', 'content': ...}, dict{'role': 'assistant', 'content': ...}]
    """
    breakpoint()
    processed_batch = []
    for row in batch:
        instruction_prompt: str = row['instruction']
        chosen_output: str = row['chosen']
        rejected_output: str = row['rejected']
        chosen = [{'role': 'user', 'content': instruction_prompt},
                    {'role': 'assistant', 'content': chosen_output}]
        rejected = [{'role': 'user', 'content': instruction_prompt},
                    {'role': 'assistant', 'content': rejected_output}]
        processed_batch.append({'chosen': chosen, 'rejected': rejected})
    return processed_batch
            
    # elif isinstance(batch, dict):
    #     """ Format is batch = {'chosen': } """
    #     instruction_prompts: list[str] = batch['instruction']
    #     chosen_outputs: list[str] = batch['chosen']
    #     rejected_outputs: list[str] = batch['rejected']
    #     dpo_generator = zip(instruction_prompts,
    #                         chosen_outputs, rejected_outputs)

    #     instructions, chosen_prompts, rejected_prompts = [], [], []
    #     for row in dpo_generator:
    #         instruction, chosen, rejected = row
    #         chosen = [{'role': 'user', 'content': instruction},
    #                   {'role': 'assistant', 'content': chosen}]
    #         rejected = [{'role': 'user', 'content': instruction},
    #                     {'role': 'assistant', 'content': rejected}]
    #         processed_batch.append({'chosen': chosen, 'rejected': rejected})
    #     ...


def main():
    dataset = load_dataset('coseal/CodeUltraFeedback_binarized', split='train')
    # dataset = dataset.map(batch_process_data, batched=True,
    #                       batch_size=2, num_proc=1)
    dataset = dataset.map(process_data, batched=False)
    breakpoint()


if __name__ == "__main__":

    main()

    # og_dataset = load_dataset('trl-lib/ultrafeedback_binarized', split='train')
    # """
    # Format of the trl-lib/ultrafeedback_binarized dataset:
    # =====
    # Each Row: ['chosen': ..., 'rejected': ..., 'score_chosen': ..., 'score_rejected': ...]

    # chosen: List[dict{'role': 'user', 'content': ...}, dict{'role': 'assistant', 'content': ...}]
    # rejected: List[dict{'role': 'user', 'content': ...}, dict{'role': 'assistant', 'content': ...}]
    # """
    # dataset = load_dataset('coseal/CodeUltraFeedback_binarized', split='train')

    breakpoint()
