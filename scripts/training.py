from sklearn.model_selection import train_test_split
from torch.utils.data import IterableDataset
import torch
import random
import numpy as np
from copy import copy
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import wandb
wandb.login()

from huggingface_hub import notebook_login
notebook_login()

with open("cleaned_dataset (1).jsonl", "r") as f:
    data = [json.loads(line) for line in f]

train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
#len(train_data), len(test_data)
#(3419, 380)

MODEL="bigcode/starcoderbase-1b"

SEQ_LENGTH=1024
BATCH_SIZE=2                   
GR_ACC_STEPS=8                   
LR=5e-5                        
LR_SCHEDULER_TYPE="cosine"       
WEIGHT_DECAY=0.01             
NUM_WARMUP_STEPS=5
EVAL_FREQ=10                    
SAVE_FREQ=10           
LOG_FREQ=10
OUTPUT_DIR="peft-starcoder-lora-a100" 
BF16=True                        
FP16=False                    

FIM_RATE=0.5                    
FIM_SPM_RATE=0.5                

LORA_R=8                         
LORA_ALPHA=32                    
LORA_DROPOUT=0.0                 
LORA_TARGET_MODULES="c_proj,c_attn,q_attn,c_fc,c_proj"    

USE_NESTED_QUANT=True           
BNB_4BIT_COMPUTE_DTYPE="bfloat16"

SEED=0

MAX_LENGTH = 1024

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    logging,
    set_seed,
    BitsAndBytesConfig,
)

set_seed(SEED)

class StreamingPromptDataset(IterableDataset):
    def __init__(
        self,
        tokenizer,
        dataset,
        max_length=1024,
        prompt_field="prompt",
        completion_field="completion",
        infinite=False,
        seed=42,
    ):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_length = max_length
        self.prompt_field = prompt_field
        self.completion_field = completion_field
        self.infinite = infinite
        self.seed = seed

    def tokenize_example(self, example):
        prompt = example["prompt"]
        completion = example["completion"]

        prompt_ids = tokenizer(prompt, add_special_tokens=False, padding="max_length", truncation=True, max_length=MAX_LENGTH).input_ids
        completion_ids = tokenizer(completion, add_special_tokens=False, padding="max_length", truncation=True, max_length=MAX_LENGTH).input_ids
        input_ids = prompt_ids + completion_ids

        labels = copy(input_ids)

        return {
            "input_ids": input_ids,
            "labels": labels
        }


    def __iter__(self):
        rng = random.Random(self.seed)
        while True:
            for example in self.dataset:
                tokenized = self.tokenize_example(example)
                if tokenized:
                    yield tokenized
            if not self.infinite:
                break


tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoderbase-1b")
tokenizer.pad_token = tokenizer.eos_token

train_dataset = StreamingPromptDataset(
    tokenizer=tokenizer,
    dataset=train_data,
    max_length=1024,
    infinite=True,
)

eval_dataset = StreamingPromptDataset(
    tokenizer=tokenizer,
    dataset=test_data,
    max_length=1024,
    infinite=False,
)


from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer

load_in_8bit = False

compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)

device_map = {"": 0}

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map=device_map,
    use_cache=False,
    trust_remote_code=True,
    use_flash_attention_2=False,  # You may want to disable this too for CPU
)

model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    r=LORA_R,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=LORA_TARGET_MODULES.split(","),
)

model = get_peft_model(model, peft_config)
#model.print_trainable_parameters() #trainable params: 5,554,176 || all params: 1,142,761,472 || trainable%: 0.4860

wandb.init(
    project="coding-llm",
    name="run-starcode-finetune",
    config={"model": MODEL, "lr": LR},
)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir=f"IvAnastasia/{OUTPUT_DIR}",
    dataloader_drop_last=True,
    evaluation_strategy="steps",
    save_strategy="steps",
    gradient_checkpointing=True,
    eval_steps=EVAL_FREQ,
    save_steps=SAVE_FREQ,
    logging_steps=LOG_FREQ,
    max_steps=50,
    logging_dir="./logs",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=LR,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    warmup_steps=NUM_WARMUP_STEPS,
    gradient_accumulation_steps=GR_ACC_STEPS,
    fp16=False,
    bf16=True,
    weight_decay=WEIGHT_DECAY,
    push_to_hub=True,
    include_tokens_per_second=True,
    save_total_limit=2,
    report_to="wandb",
    run_name="run-starcode-finetune",
)


model.train()
for param in model.parameters():
    param.requires_grad = True


trainer = Trainer(
    model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset
)

print("Training...")
trainer.train()
