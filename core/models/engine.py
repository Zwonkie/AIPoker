import os
import torch
from core.models.poker_transformer import PokerEVModelV4

class ModelEngine:
    """
    Wrapper around pure PyTorch neural networks for inference.
    Exclusively manages V8/V9 model architectures (PokerEVModelV4) after older models were pruned.
    """
    def __init__(self, expert_name: str, device: str = 'cpu', is_v11: bool = False):
        self.device = torch.device(device)
        self.is_v11 = is_v11
        
        if self.is_v11:
            from core.models.v11.poker_transformer_v11 import PokerEVModelV4 as PokerEVModelV11
            self.model = PokerEVModelV11().to(self.device)
        else:
            self.model = PokerEVModelV4().to(self.device)
        
        # weights are in core/weights
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        w_dir = os.path.join(base_dir, 'weights')
        
        weight_path = os.path.join(w_dir, expert_name)
        try:
            checkpoint = torch.load(weight_path, map_location=self.device, weights_only=True)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.eval()
        except FileNotFoundError:
            print(f"WARNING: {expert_name} not found at {weight_path}. Model will output random garbage.")
        except RuntimeError as e:
            print(f"WARNING: Could not load weights: {e}")

    def predict_ev(self, hole: torch.Tensor, board: torch.Tensor, ctx: torch.Tensor, act: torch.Tensor) -> dict:
        """
        Runs the forward pass on the PyTorch model.
        Returns the EVs for Fold, Call, Raise for the final sequence step.
        """
        with torch.no_grad():
            q_vals = self.model(hole.to(self.device), board.to(self.device), ctx.to(self.device), act.to(self.device))
        
        # Q-values shape: [batch, seq_len, 3]
        # We only care about the Q-values of the final state step
        final_q = q_vals[0, -1, :].cpu().numpy()
        return {'FOLD': float(final_q[0]), 'CALL': float(final_q[1]), 'RAISE': float(final_q[2])}
