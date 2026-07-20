import torch
import torch.nn as nn
import math

class PokerEVModelV4(nn.Module):
    """
    Decision Transformer model for Pluribus V4.
    Processes sequences of (State, Action) inputs using Multi-Head Self-Attention.
    """
    def __init__(self, card_embed_dim=16, context_dim=44, d_model=128, nhead=4, num_layers=3, dim_feedforward=512, max_seq_len=20, num_actions=6):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        # V14: discretized bet-size action space -> [fold, call, raise_33, raise_66, raise_pot, allin].
        # (V13 was 3: fold/call/raise.) Both critic (q_vals) and actor (policy_logits) heads are K-wide.
        self.num_actions = num_actions
        # Diagnostic: when True the hole-card embedding is zeroed, forcing the model to
        # derive hand strength from equity+board+context instead of memorizing hole ranks.
        # Used to test whether equity can be made load-bearing (no tensor-shape change, so
        # checkpoints stay compatible). Default False = normal behavior.
        self.ablate_hole_cards = False
        
        # 1. Embeddings
        self.card_embedding = nn.Embedding(num_embeddings=53, embedding_dim=card_embed_dim, padding_idx=52)
        
        # Action vocabulary size: 10 (Pad, Bet, Check, Raise, Fold, etc.)
        self.action_embedding = nn.Embedding(num_embeddings=10, embedding_dim=d_model, padding_idx=0)
        
        # State Projection: maps raw card sum + context features into d_model space
        # Card representation is: hole cards (64-dim) + board cards (64-dim) = 128-dim
        # Context representation is: 35-dim
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
        
        # 3. Critic Head: predicts per-action counterfactual values Q [Fold, Call, Raise].
        # In V12 this is the CRITIC only — it is trained to regress the simulator's
        # all-action Monte-Carlo counterfactual EVs, and it feeds the policy target during
        # training. It is NO LONGER argmax'd to pick actions (that was the V11 failure mode).
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

        # 3b. Actor Head (V12): outputs an action DISTRIBUTION over [Fold, Call, Raise].
        # Trained toward a regret-matching policy computed from the counterfactual action
        # values, so decisions come from a normalized distribution instead of an argmax over
        # one uncalibrated Q head. This structurally removes the raise-/call-everything
        # degeneracy (a single over-estimated head can no longer capture every decision).
        self.head_policy = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

        # 4. Auxiliary Heads (Interpretable Subconscious)
        self.head_bluff = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_strength = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_equity = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))

        # 5. EQUITY-PRIMARY base heads. Q and policy are structured as
        #        value = base(equity, price) + residual(transformer over cards/board).
        # The base path sees ONLY strength+price scalars (equity, pot_odds, pot, call, street),
        # so hand strength MUST flow through equity -- it cannot be replaced by a hole-card
        # lookup. The card/board transformer is bottlenecked (card_embed_dim=16) and only adds
        # a refinement (draws, blockers, texture). Combined with the zero-init of the residual
        # heads below, the model STARTS as a pure-equity player and learns cards as corrections.
        # context indices for the base head: equity, pot_odds, pot, call, street.
        # (Opponent-tendency features 7,8 were tried here but HURT — the model needs adaptation
        # via range-aware equity, not raw HUD scalars bolted onto the base head.)
        self.SP_IDX = [3, 4, 2, 9, 6]
        self.equity_base_q = nn.Sequential(nn.Linear(len(self.SP_IDX), 32), nn.ReLU(), nn.Linear(32, num_actions))
        self.equity_base_pi = nn.Sequential(nn.Linear(len(self.SP_IDX), 32), nn.ReLU(), nn.Linear(32, num_actions))
        # Zero-init the residual heads' final layer so value == base at init (equity-primary).
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
        nn.init.zeros_(self.head_policy[-1].weight); nn.init.zeros_(self.head_policy[-1].bias)
        
    def _generate_causal_mask(self, sz, device):
        """Generate a standard causal mask (future states are True, which means masked out)."""
        return torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, hole, board, context, actions, key_padding_mask=None):
        """
        Args:
            hole: [batch, 2] (integer card indices)
            board: [batch, seq_len, 5] (integer card indices)
            context: [batch, seq_len, 31] (numerical features)
            actions: [batch, seq_len] (historical action indices)
            key_padding_mask: [batch, seq_len] (ignored/retained for compatibility)
        Returns:
            dict with 'q_vals' [batch, seq_len, 3] (critic), 'policy_logits'
            [batch, seq_len, 3] (actor), and the aux heads.
        """
        batch_size, seq_len, _ = context.shape
        device = context.device
        
        # 1. Embed Hole Cards
        # hole_emb: [batch, 2, 64] -> sum to [batch, 64] -> expand to [batch, seq_len, 64]
        hole_emb = self.card_embedding(hole).sum(dim=1).unsqueeze(1).expand(-1, seq_len, -1)
        if self.ablate_hole_cards:
            hole_emb = torch.zeros_like(hole_emb)
        
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
        transformer_out = self.transformer(x, mask=mask, src_key_padding_mask=key_padding_mask) # [batch, seq_len, 128]
        
        # 7. Predict critic Q-values and actor policy logits as EQUITY-PRIMARY base + residual.
        # Base sees only strength+price scalars (so equity is load-bearing); the transformer
        # residual refines with card/board detail.
        sp = context[:, :, self.SP_IDX]                     # [batch, seq_len, 5]
        q_vals = self.equity_base_q(sp) + self.head(transformer_out)             # critic
        policy_logits = self.equity_base_pi(sp) + self.head_policy(transformer_out)  # actor

        return {
            'q_vals': q_vals,
            'policy_logits': policy_logits,
            'bluff': self.head_bluff(transformer_out).squeeze(-1),
            'strength': self.head_strength(transformer_out).squeeze(-1),
            'equity': self.head_equity(transformer_out).squeeze(-1)
        }
