import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout_rate):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True  # if PyTorch >= 1.10
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.layernorm1 = nn.LayerNorm(hidden_dim)
        self.layernorm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_mask=None):
        # Self-Attention
        attn_out, _ = self.attn(
            x,  # queries
            x,  # keys
            x,  # values
            attn_mask=attn_mask
        )
        # Residual + layer norm
        x = self.layernorm1(x + self.dropout(attn_out))
        
        # Feed-Forward
        ff_out = self.feed_forward(x)
        
        # Residual + layer norm
        x = self.layernorm2(x + self.dropout(ff_out))
        
        return x



class SASRec(nn.Module):
    def __init__(
        self,
        num_items,
        hidden_dim=64,
        max_seq_len=50,
        num_heads=2,
        num_blocks=2,      # <-- variable number of blocks
        dropout_rate=0.2
    ):
        super().__init__()
        
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        # Item + positional embeddings
        self.item_embedding = nn.Embedding(num_items, hidden_dim, padding_idx=0)
        self.pos_embedding  = nn.Embedding(max_seq_len, hidden_dim)
        
        # Build a list of Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout_rate)
            for _ in range(num_blocks)
        ])
        
        # Final linear for your task. 
        # E.g., a single logit for binary classification:
        self.output_layer = nn.Linear(hidden_dim, 1)
    
    def forward(self, seq):
        """
        seq: [B, S] of item IDs
        returns [B, S, hidden_dim] - hidden representation after all blocks
        """
        B, S = seq.size()
        
        # Position indices [0..S-1]
        positions = torch.arange(S, device=seq.device).unsqueeze(0)  # [1, S]
        
        # Embeddings
        seq_emb = self.item_embedding(seq)          # [B, S, hidden_dim]
        pos_emb = self.pos_embedding(positions)     # [1, S, hidden_dim]
        x = seq_emb + pos_emb                       # initial input
        
        # Causal mask: each position can only attend to <= that position
        # shape: [S, S], True => block
        causal_mask = torch.ones((S, S), device=seq.device).triu(1).bool()
        
        # Pass through each Transformer block in sequence
        for block in self.blocks:
            x = block(x, attn_mask=causal_mask)
        
        return x  # [B, S, hidden_dim]
    
    def predict_logit(self, seq_repr, target_item):
        """
        Given a final seq_repr [B, hidden_dim] and a target_item [B],
        returns a logit for binary classification (BCEWithLogitsLoss).
        """
        target_emb = self.item_embedding(target_item)    # [B, hidden_dim]
        
        # E.g., simple elementwise product + linear
        dot_prod = (seq_repr * target_emb).sum(dim=-1, keepdim=True)  # [B, 1]
        # logit = self.output_layer(dot_prod)                            # [B, 1]
        return dot_prod.squeeze(-1)                                       # [B]