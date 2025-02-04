import numpy as np
import pandas as pd
import re
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.CONSTANTS import DATA_PATH, PROMPT
from utils.calculate_metrics import binary_classification_metrics

FEW_SHOT = None

## Load Processed Data
train_data = pd.read_csv(DATA_PATH + '/train_data_it.csv')
test_data = pd.read_csv(DATA_PATH + '/test_data_it.csv')


## Train data for Few-shot examples
train_data = train_data.head(10)


## Convert string to list
train_data['past_movies'] = train_data['past_movies'].apply(eval)
test_data['past_movies'] = test_data['past_movies'].apply(eval)

# Create a Hugging Face dataset from the sampled_movie_df DataFrame
test_dataset = Dataset.from_pandas(test_data)
print(test_dataset)


## Generate Prompts
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


## Create few-shot examples
def create_few_shots(prompt, past_movies, candidate, rating):
    for movie in past_movies:
        title, genre, r = movie.split(":::")
        if int(r) > 3.0:
            prompt += f'- Liked "{title}"\n'
        else:
            prompt += f'- Disliked "{title}"\n'
    target_movie_title, target_movie_genre = candidate.split(":::")
    prompt += f'\n## Target movie:\n"{target_movie_title}"\n'
    prompt += f'\n## Response:\n'
    if float(rating) > 3.0:
        prompt += "$\\boxed{Yes}$"
    else:
        prompt += '$\\boxed{No}$'
    return prompt

if FEW_SHOT != None:
    EXAMPLES = ""
    prompt = "### User's rating history\n"
    for i in range(FEW_SHOT):
        x = train_data.iloc[i]
        EXAMPLES += f"## Example {i+1}\n"
        EXAMPLES += create_few_shots(prompt, x['past_movies'], x['candidate'], x['rating'])
        EXAMPLES += "\n\n"
    EXAMPLES += "## Now, do the same for the following user\n"
    # Add Examples to the Prompt
    PROMPT += EXAMPLES
    PROMPT += "## User’s history:\n"

## Print system instruction
print("System Prompt:")
print(PROMPT)
print("="*50)

## Check if MPS is available
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device: {device}")

## Load model and tokenizer from HuggingFace.
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use float16 for efficiency
)
model = model.to(device)
print("Model loaded successfully!")

## Generate responses
print("generating responses for the test set...")
actuals = []
preds = []
for i in range(5):
    x = test_dataset[i]
    prompt = create_prompt(PROMPT, x['past_movies'], x['candidate'])
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        num_return_sequences=1
    )
    res = tokenizer.decode(outputs[0], skip_special_tokens=True)
    match = re.search(r'\\boxed{(.+?)}', res)
    # print(res)
    if match:
        preds.append(match.group(1))
        actuals.append(int(x['rating']))
    else:
        continue
    if i % 500 == 0:
        print(f"Generated {i} responses...")

## Compute classification metrics
binary_classification_metrics(actuals, preds)