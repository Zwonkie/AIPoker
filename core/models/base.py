from abc import ABC, abstractmethod

class PokerModelInterface(ABC):
    @abstractmethod
    def predict_action(self, board: list, hand: list, equity: float, pot_size: float, 
                       call_amount: float, hero_stack: float, num_opponents: int,
                       is_preflop: bool, use_preflop_chart: bool, use_math_engine: bool,
                       use_bluff_engine: bool, use_dynamic_sizing: bool,
                       bet_raise_available: bool, check_call_available: bool,
                       active_opponents: list = None) -> tuple:
        """
        Predicts action, returning: (action, reason, bet_size)
        """
        pass
