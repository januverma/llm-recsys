import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.CONSTANTS import DATA_PATH, MAX_MOVIE_SEQ_LENGTH, SASREC_CONFIG
from utils.pytorch_data_utils import ImplicitDataset, collate_fn
from utils.sasrec import SASRec
from utils.calculate_metrics import binary_classification_metrics

## Constants
BATCH_SIZE = SASREC_CONFIG["BATCH_SIZE"]
HIDDEN_DIM = SASREC_CONFIG["HIDDEN_DIM"]
NUM_HEADS = SASREC_CONFIG["NUM_HEADS"]
NUM_BLOCKS = SASREC_CONFIG["NUM_BLOCKS"]
DROPOUT = SASREC_CONFIG["DROPOUT"]
LR = SASREC_CONFIG["LR"]
EPOCHS = SASREC_CONFIG["EPOCHS"]


## Load Processed Data
train_data = pd.read_csv(DATA_PATH + '/train_data_it.csv')
test_data = pd.read_csv(DATA_PATH + '/test_data_it.csv')

## Add label column (binary)
train_data['label'] = (train_data['rating'] > 3).astype(int)
test_data['label'] = (test_data['rating'] > 3).astype(int)

## Convert string to list
train_data['past_movie_ids'] = train_data['past_movie_ids'].apply(eval)
test_data['past_movie_ids'] = test_data['past_movie_ids'].apply(eval)


## Create DataLoader
dataset = ImplicitDataset(train_data)
loader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    collate_fn=lambda b: collate_fn(b, max_seq_len=MAX_MOVIE_SEQ_LENGTH)
)

## Check the first batch
print("Check the first batch")
for batch_seq, batch_target, batch_label in loader:
    print(batch_seq.shape, batch_target.shape, batch_label.shape)
    break

## Check if MPS is available
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device: {device}")


# You need the total number of items to size the embedding. 
# Typically, you’d find the max item ID and add 1 (for 0-based indexing).
# For a quick hack, let's assume:
max_item_id = max(train_data['movieId'].max(), max(max(seq) for seq in train_data['past_movie_ids'] if len(seq)>0))
num_items = max_item_id + 1  # 0..max_item_id

# Instantiate the model
model = SASRec(
    num_items=num_items,
    hidden_dim=HIDDEN_DIM,
    max_seq_len=MAX_MOVIE_SEQ_LENGTH,
    num_heads=NUM_HEADS,
    num_blocks=NUM_BLOCKS,  # <-- 2 blocks
    dropout_rate=DROPOUT
).to(device)

print("Initialized SASRec model with config: \n")
print(f"HIDDEN_DIM: {HIDDEN_DIM}, NUM_HEADS: {NUM_HEADS}, NUM_BLOCKS: {NUM_BLOCKS}, DROPOUT: {DROPOUT}")

# Define loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Train the model
print(f"Start training {EPOCHS} epochs...")
model.train()

for epoch in range(EPOCHS):
    total_loss = 0.0
    for batch_seq, batch_target, batch_label in loader:
        batch_seq, batch_target, batch_label = batch_seq.to(device), batch_target.to(device), batch_label.to(device)
        optimizer.zero_grad()
        
        # 1) Forward pass: get the sequence hidden states
        all_hidden = model(batch_seq)           # [B, S, hidden_dim]
        seq_repr   = all_hidden[:, -1, :]       # use the last position as user representation
        
        # 2) Predict logit for (seq_repr, target_item)
        logit = model.predict_logit(seq_repr, batch_target)  # [B]
        
        # 3) Compute BCE loss with logits
        loss = criterion(logit, batch_label)
        
        # 4) Backprop
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss = {avg_loss:.4f}")


## Evaluate the model
### Prepare Test DataLoader
test_dataset = ImplicitDataset(test_data)
test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,  # typically don't need shuffle for test
    collate_fn=lambda b: collate_fn(b, max_seq_len=MAX_MOVIE_SEQ_LENGTH)
)

### Evaluate
model.eval()  # switch to evaluation mode (no dropout, no grad, etc.)

all_preds = []
all_labels = []

with torch.no_grad():
    for batch_seq, batch_target, batch_label in test_loader:
        batch_seq, batch_target, batch_label = batch_seq.to(device), batch_target.to(device), batch_label.to(device)
        # 1) Forward pass: get the last hidden state
        all_hidden = model(batch_seq)         # [B, S, hidden_dim]
        seq_repr   = all_hidden[:, -1, :]     # user representation
        
        # 2) Predict logit
        logit = model.predict_logit(seq_repr, batch_target)  # shape [B]
        
        # 3) Convert logit -> probability
        prob = torch.sigmoid(logit)  # [B]
        
        # 4) Predicted label: 1 if prob>0.5 else 0
        preds = (prob > 0.5).float()  # [B]
        
        # 5) Store for metric calculation
        all_preds.append(preds.cpu().numpy())     # or .tolist()
        all_labels.append(batch_label.cpu().numpy())

# Concatenate all batches
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

# Compute metrics
binary_classification_metrics(all_labels, all_preds)