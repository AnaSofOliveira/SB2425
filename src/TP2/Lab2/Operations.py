from enum import Enum

class LogicalOperations(Enum):
    AND = 1
    OR = 2
    XOR = 3
    NOT = 4

class ArtihmeticOperations(Enum):
    ADD = 1
    SUBTRACT = 2
    MULTIPLY = 3
    DIVIDE = 4

class SharpeningOperations(Enum):
    CANNY = 1
    CUSTOM = 2
    SOBEL = 3
    ROBERTS = 4
    PREWITT = 5
    LOG = 6