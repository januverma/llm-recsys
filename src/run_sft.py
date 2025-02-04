import pandas as pd
import numpy as np
import torch

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.CONSTANTS import DATA_PATH, MAX_MOVIE_SEQ_LENGTH
from utils.SFT_CONFIG import SFT_PROMPT, LORA_CONFIG, MODEL_NAME
from utils.calculate_metrics import binary_classification_metrics

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import unsloth_train

## Load Processed Data
train_data = pd.read_csv(DATA_PATH + '/train_data_it.csv')
test_data = pd.read_csv(DATA_PATH + '/test_data_it.csv')

# Instruction Data
from datasets import Dataset

# Create a Hugging Face dataset from the train and test DataFrame
train_dataset = Dataset.from_pandas(train_data)
print(train_dataset)

# Create a Hugging Face dataset from the sampled_movie_df DataFrame
test_dataset = Dataset.from_pandas(test_data)
print(test_dataset)

# Evaluate the model on a small sample of the evaluation dataset
val_sample_size = 1000  # Adjust the sample size as needed
val_dataset = test_dataset.select(range(min(val_sample_size, len(test_dataset))))

# Unsloth Configuration
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# Load the model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = LORA_CONFIG['r'],
    target_modules = LORA_CONFIG['target_modules'],
    lora_alpha = LORA_CONFIG['lora_alpha'], 
    lora_dropout = LORA_CONFIG['lora_dropout'], # Supports any, but = 0 is optimized
    bias = LORA_CONFIG['bias'],    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = LORA_CONFIG['use_gradient_checkpointing'], # True or "unsloth" for very long context
    random_state = LORA_CONFIG['random_state'], # Set this for reproducibility
    use_rslora = LORA_CONFIG['user_rslora'],  # We support rank stabilized LoRA
    loftq_config = LORA_CONFIG['loftq_config'], # And LoftQ
)

def create_prompt(prompt, past_movies, candidate):
    for movie in past_movies:
        title, genre, rating = movie.split(":::")
        if int(rating) > 3.0:
            prompt += f'- Liked "{title}"\n'
        else:
            prompt += f'- Disliked "{title}"\n'
    target_movie_title, target_movie_genre = candidate.split(":::")
    prompt += f'\n## Target movie:\n"{target_movie_title}"\n'
    prompt += f'\n## Response:\n'
    return prompt

# Add the prompt to the dataset
EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = [SFT_PROMPT] * len(examples["userId"])
    userIds       = examples["userId"]
    inputs       = examples["past_movies"]
    candidates = examples["candidate"]
    outputs      = examples["rating"]
    texts = []
    for instruction, userId, input, candidate, output in zip(instructions, userIds, inputs, candidates, outputs):
        instruction = create_prompt(instruction, input, candidate)
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        if int(output) > 3:
          instruction += 'Yes'
        else:
          instruction += 'No'
        instruction += EOS_TOKEN
        texts.append(instruction)
    return { "text" : texts, }
pass

# Apply the formatting function to the train and validation datasets
train_dataset = train_dataset.map(formatting_prompts_func, batched = True,)
val_dataset = val_dataset.map(formatting_prompts_func, batched = True,)

# Trainer Arguments
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = val_dataset, # Added eval_dataset
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 64,
        gradient_accumulation_steps = 1,
        warmup_steps = 20,
        num_train_epochs = 1, # Set this for 1 full training run.
        # max_steps = 200,
        learning_rate = 2e-5,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
        eval_strategy="steps", # Added evaluation strategy
        eval_steps=50, # Evaluate every 50 steps
    ),
)

## Train
trainer_stats = unsloth_train(trainer)
print(trainer_stats)

## Evaluate
FastLanguageModel.for_inference(model)
actuals = []
preds = []
for i in range(len(test_dataset)):
    x = test_dataset[i]
    inputs = tokenizer(
        [
            create_prompt(SFT_PROMPT, x['past_movies'], x['candidate'])
        ], return_tensors = "pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens = 1, use_cache = True)
    res = tokenizer.batch_decode(outputs)[0]
    try:
        preds.append(res.split("## Response:\n")[-1].strip())
        actuals.append(int(x['rating']))
    except:
        continue

# Calculate Metrics
binary_classification_metrics(actuals, preds)


