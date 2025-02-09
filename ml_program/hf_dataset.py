""" Processing dataset for coding datasets """

from __future__ import annotations
from datasets import Dataset, load_from_disk, load_dataset
from enum import Enum

class HFDatasets: 
    CODEFORCE_BIN='coseal/CodeUltraFeedback_binarized'
    """Link: https://huggingface.co/datasets/coseal/CodeUltraFeedback_binarized"""  
    
class HFModels: 
    QWEN="Qwen/Qwen2-0.5B-Instruct"
    
    
    
# TODO: pre-process the dataset so that it matches this one right here 
# MUST MATCH THIS FORMAT: https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized?row=0
def batch_process_data(batch): 
    ...
    
if __name__=="__main__": 
    
    
    breakpoint()
    