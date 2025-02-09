
from __future__ import annotations
from datasets import load_dataset, Dataset


def split_prompt_and_response(conversations):
    """Extracts prompts and responses from a batch of conversations."""
    prompts = []
    responses = []
    for conversation in conversations:
        prompt = []
        response = []
        for msg in conversation:
            if msg["role"] == "user":
                prompt.append(msg["content"])
            else:
                response.append(msg["content"])
        prompts.append("\n".join(prompt))
        responses.append("\n".join(response))
    return prompts, responses

def preprocess_function(examples):
    # Use the fixed length from your training config (256)
    fixed_length = train_config.trainer_config.max_length  # e.g., 256

    # Separate prompts and responses for 'chosen' and 'rejected'
    prompts_chosen, responses_chosen = split_prompt_and_response(examples["chosen"])
    prompts_rejected, responses_rejected = split_prompt_and_response(examples["rejected"])

    # Tokenize in batches using the fixed length (256)
    tokenized_prompts = tokenizer(prompts_chosen, max_length=fixed_length, truncation=True, padding='max_length')
    tokenized_chosen = tokenizer(responses_chosen, max_length=fixed_length, truncation=True, padding='max_length')
    tokenized_rejected = tokenizer(responses_rejected, max_length=fixed_length, truncation=True, padding='max_length')

    return {
        "prompt_input_ids": tokenized_prompts["input_ids"],
        "prompt_attention_mask": tokenized_prompts["attention_mask"],
        "chosen_input_ids": tokenized_chosen["input_ids"],
        "chosen_attention_mask": tokenized_chosen["attention_mask"],
        "rejected_input_ids": tokenized_rejected["input_ids"],
        "rejected_attention_mask": tokenized_rejected["attention_mask"],
        "score_chosen": examples["score_chosen"],
        "score_rejected": examples["score_rejected"],
    }
    
    
if __name__=="__main__":    
    # Apply the preprocess function in batched mode:
    train_dataset = load_dataset()
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    
    
    prompt_lengths = [len(sample["prompt_input_ids"]) for sample in train_dataset]
    chosen_lengths = [len(sample["chosen_input_ids"]) for sample in train_dataset]
    rejected_lengths = [len(sample["rejected_input_ids"]) for sample in train_dataset]

    print("Unique prompt lengths:", set(prompt_lengths))
    print("Unique chosen lengths:", set(chosen_lengths))
    print("Unique rejected lengths:", set(rejected_lengths))
    
    
    # def split_prompt_and_response(conversations):
    #     """Extracts prompts and responses from a batch of conversations."""
    #     prompts = []
    #     responses = []
    #     for conversation in conversations:
    #         prompt = []
    #         response = []
    #         for msg in conversation:
    #             if msg["role"] == "user":
    #                 prompt.append(msg["content"])
    #             else:
    #                 response.append(msg["content"])
    #         prompts.append("\n".join(prompt))
    #         responses.append("\n".join(response))
    #     return prompts, responses

    # def preprocess_function(examples):
    #     # Use the fixed length from your training config (256)
    #     fixed_length = train_config.trainer_config.max_length  # e.g., 256

    #     # Separate prompts and responses for 'chosen' and 'rejected'
    #     prompts_chosen, responses_chosen = split_prompt_and_response(examples["chosen"])
    #     prompts_rejected, responses_rejected = split_prompt_and_response(examples["rejected"])

    #     # Tokenize in batches using the fixed length (256)
    #     tokenized_prompts = tokenizer(prompts_chosen, max_length=fixed_length, truncation=True, padding='max_length')
    #     tokenized_chosen = tokenizer(responses_chosen, max_length=fixed_length, truncation=True, padding='max_length')
    #     tokenized_rejected = tokenizer(responses_rejected, max_length=fixed_length, truncation=True, padding='max_length')

    #     return {
    #         "prompt_input_ids": tokenized_prompts["input_ids"],
    #         "prompt_attention_mask": tokenized_prompts["attention_mask"],
    #         "chosen_input_ids": tokenized_chosen["input_ids"],
    #         "chosen_attention_mask": tokenized_chosen["attention_mask"],
    #         "rejected_input_ids": tokenized_rejected["input_ids"],
    #         "rejected_attention_mask": tokenized_rejected["attention_mask"],
    #         "score_chosen": examples["score_chosen"],
    #         "score_rejected": examples["score_rejected"],
    #     }
    # # Apply the preprocess function in batched mode:
    # train_dataset = train_dataset.map(preprocess_function, batched=True)