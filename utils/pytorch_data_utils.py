import torch
from torch.utils.data import Dataset

## Define the Dataset class
class ImplicitDataset(Dataset):
    def __init__(self, df):
        """
        df must have columns: 
          - 'past_movie_ids': List of item IDs
          - 'movieId':        The target item
          - 'label':          0 or 1
        """
        self.past_seqs = df['past_movie_ids'].tolist()
        self.targets   = df['movieId'].tolist()
        self.labels    = df['label'].tolist()
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        seq = self.past_seqs[idx]
        target = self.targets[idx]
        label = self.labels[idx]
        return seq, target, label
    

def collate_fn(batch, max_seq_len=50):
    """
    batch: list of (seq, target, label) from __getitem__()
    """
    seqs, targets, labels = zip(*batch)
    batch_size = len(seqs)
    
    # Prepare padded sequences
    padded_seqs = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    
    for i, seq in enumerate(seqs):
        # Truncate (from the left) if seq is too long
        trunc_seq = seq[-max_seq_len:]
        seq_len = len(trunc_seq)
        padded_seqs[i, :seq_len] = torch.tensor(trunc_seq, dtype=torch.long)
        
    targets = torch.tensor(targets, dtype=torch.long)
    labels  = torch.tensor(labels, dtype=torch.float)  # float for BCE
    
    return padded_seqs, targets, labels