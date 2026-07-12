import torch
import torch.nn as nn
import math

class PokerEVModelV4(nn.Module):
    """
    Decision Transformer model for Pluribus V4.
    Processes sequences of (State, Action) inputs using Multi-Head Self-Attention.
    """
    def __init__(self, card_embed_dim=64, context_dim=31, d_model=128, nhead=4, num_layers=3, dim_feedforward=512, max_seq_len=20):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # 1. Embeddings
        self.card_embedding = nn.Embedding(num_embeddings=53, embedding_dim=card_embed_dim, padding_idx=52)
        
        # Action vocabulary size: 10 (Pad, Bet, Check, Raise, Fold, etc.)
        self.action_embedding = nn.Embedding(num_embeddings=10, embedding_dim=d_model, padding_idx=0)
        
        # State Projection: maps raw card sum + context features into d_model space
        # Card representation is: hole cards (64-dim) + board cards (64-dim) = 128-dim
        # Context representation is: 31-dim
        self.state_proj = nn.Sequential(
            nn.Linear(card_embed_dim * 2 + context_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, d_model)
        )
        
        # Positional Embeddings
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        # 2. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Output Head: predicts Q-values [Fold, Call, Raise] for each step
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
    def _generate_causal_mask(self, sz, device):
        """Generate a mask that isolates each step (only diagonal is False/allowed) to prevent padding dilution and avoid NaNs."""
        return ~torch.eye(sz, dtype=torch.bool, device=device)

    def forward(self, hole, board, context, actions, key_padding_mask=None):
        """
        Args:
            hole: [batch, 2] (integer card indices)
            board: [batch, seq_len, 5] (integer card indices)
            context: [batch, seq_len, 31] (numerical features)
            actions: [batch, seq_len] (historical action indices)
            key_padding_mask: [batch, seq_len] (ignored/retained for compatibility)
        Returns:
            Q-values: [batch, seq_len, 3]
        """
        batch_size, seq_len, _ = context.shape
        device = context.device
        
        # 1. Embed Hole Cards
        # hole_emb: [batch, 2, 64] -> sum to [batch, 64] -> expand to [batch, seq_len, 64]
        hole_emb = self.card_embedding(hole).sum(dim=1).unsqueeze(1).expand(-1, seq_len, -1)
        
        # 2. Embed Board Cards at each step
        # board_emb: [batch, seq_len, 5, 64] -> sum to [batch, seq_len, 64]
        board_emb = self.card_embedding(board).sum(dim=2)
        
        # 3. Concatenate and Project to State Embeddings
        # concat: [batch, seq_len, 64 + 64 + 31] = [batch, seq_len, 159]
        state_features = torch.cat([hole_emb, board_emb, context], dim=2)
        state_emb = self.state_proj(state_features) # [batch, seq_len, 128]
        
        # 4. Embed Actions (Shifted by 1 to represent PrevAction)
        # We prepend a padding action at index 0 for the first step
        shifted_actions = torch.zeros_like(actions)
        shifted_actions[:, 1:] = actions[:, :-1]
        act_emb = self.action_embedding(shifted_actions) # [batch, seq_len, 128]
        
        # 5. Form Transformer Inputs: State + PrevAction + Positional Encoding
        x = state_emb + act_emb + self.pos_emb[:, :seq_len, :]
        
        # 6. Process sequence with step-isolating mask (no dilution, no NaNs)
        mask = self._generate_causal_mask(seq_len, device)
        transformer_out = self.transformer(x, mask=mask) # [batch, seq_len, 128]
        
        # 7. Predict Q-Values
        q_vals = self.head(transformer_out) # [batch, seq_len, 3]
        return q_vals
