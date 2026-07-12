from abc import ABC, abstractmethod
import torch
from core.board_state import BoardState
from typing import Tuple

class DataContract(ABC):
    """
    Abstract Base Class for translating pure mathematical BoardState
    into the exact tensor inputs expected by a specific Neural Network version.
    """
    
    @abstractmethod
    def to_tensors(self, board_state: BoardState) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            hole_tensor: [Batch, Sequence=2]
            board_tensor: [Batch, Sequence, 5]
            ctx_tensor: [Batch, Sequence, ContextFeatures]
            act_tensor: [Batch, Sequence]
        """
        pass
