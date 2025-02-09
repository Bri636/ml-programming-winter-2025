""" Main Script for training """
from __future__ import annotations
from typing import Union, Any
import torch
from datasets import load_dataset
from trl import CPOConfig, DPOConfig, CPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import Field
from argparse import ArgumentParser
from transformers import BitsAndBytesConfig
import transformers
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig
from transformers import DataCollatorWithPadding
# from trl.trainer.cpo_trainer import DPODataCollatorWithPadding
# imports
from ml_program.hf_dataset import HFDatasets
from ml_program.utils import BaseConfig
from ml_program.collators import DPODataCollatorWithPadding

TrainerConfig = Union[CPOConfig, DPOConfig]

_DTYPES={
    'bfloat16': torch.bfloat16
}

class QuantizationConfig(BaseConfig):
    load_in_4bit: bool = False
    bnb_4bit_quant_type: str = 'nf4'
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = 'bfloat16'

class LoraParamConfig(BaseConfig):
    base_model_name_or_path: str = "Qwen/Qwen2-0.5B-Instruct"
    r: int = 4
    lora_alpha: int = 16
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj",
                                 "v_proj", "out_proj", "fc_in", "fc_out", "wte"]
    )
    lora_dropout: float = 0.1
    bias: str = "none"
    # https://discuss.huggingface.co/t/task-type-parameter-of-loraconfig/52879/6
    task_type: str = "CAUSAL_LM"

class TrainConfig(BaseConfig):
    """ Master train config """
    model_name_or_path: str = Field(default="Qwen/Qwen2-0.5B-Instruct")
    dataset_name_or_path: str = Field(
        default='coseal/CodeUltraFeedback_binarized')
    checkpoint_every_n: int = Field(default=2)
    trainer_config: TrainerConfig = Field(default_factory=lambda: CPOConfig(
        output_dir='/lus/eagle/projects/FoundEpidem/bhsu/2024_research/ml-programming-winter-2025/checkpoints'))
    quantization_config: QuantizationConfig = Field(
        default_factory=QuantizationConfig)
    lora_config: LoraParamConfig = Field(
        default_factory=LoraParamConfig)

def parse_arguments():
    argparser = ArgumentParser()
    argparser.add_argument('--model_name_or_path',
                           type=str, default='Qwen/Qwen2-0.5B-Instruct')
    argparser.add_argument('--dataset_name_or_path',
                           type=str, default='coseal/CodeUltraFeedback_binarized')
    argparser.add_argument('--checkpoint_dir',
                           type=str, default='/lus/eagle/projects/FoundEpidem/bhsu/2024_research/ml-programming-winter-2025/checkpoints')
    argparser.add_argument('--train_config_path',
                           type=str, default='/lus/eagle/projects/FoundEpidem/bhsu/2024_research/ml-programming-winter-2025/ml_program/config_files/polaris_train_config.yaml')
    return argparser.parse_args()

def main():
    args = parse_arguments()
    # modify some fields based on arguments
    train_config = TrainConfig.from_yaml(args.train_config_path)
    # train_config = TrainConfig()
    train_config.model_name_or_path = args.model_name_or_path
    train_config.dataset_name_or_path = args.dataset_name_or_path
    train_config.trainer_config.output_dir = args.checkpoint_dir
    # workaround for dtypes and eval 
    train_config.quantization_config.bnb_4bit_compute_dtype = _DTYPES.get(train_config.quantization_config.bnb_4bit_compute_dtype)
    quantization_config = transformers.BitsAndBytesConfig(
        **train_config.quantization_config.model_dump())
    train_config.lora_config.base_model_name_or_path = args.model_name_or_path
    lora_config = LoraConfig(**train_config.lora_config.model_dump())

    train_dataset = load_dataset('trl-lib/ultrafeedback_binarized',
                                 split="train",
                                 trust_remote_code=True)

    # load in models and train
    model_base = AutoModelForCausalLM.from_pretrained(train_config.model_name_or_path,
                                                      quantization_config=quantization_config,
                                                      trust_remote_code=True, 
                                                      device_map={'':torch.cuda.current_device()})
    # https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1179
    model_base = prepare_model_for_kbit_training(model_base, use_gradient_checkpointing=False)
    model = get_peft_model(model_base, lora_config)
    # Freeze non-LoRA parameters
    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.dtype}")
    
    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name_or_path,
                                              max_length=train_config.trainer_config.max_length, 
                                              padding="max_length",
                                              truncation=True,
                                              trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    ###
    data_collator = DPODataCollatorWithPadding(pad_token_id=tokenizer.pad_token_id, 
                                               max_length=train_config.trainer_config.max_length
                                               )

    trainer = CPOTrainer(model=model,
                         args=train_config.trainer_config,
                         processing_class=tokenizer,
                         train_dataset=train_dataset, 
                         data_collator=data_collator
                         )
    trainer.train()

if __name__ == "__main__":

    main()
