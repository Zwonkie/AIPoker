from enum import Enum

class PokerAction(Enum):
    FOLD = 0
    CALL = 1
    RAISE = 2
    CHECK = 3
    ALL_IN = 4
