import enum


class RewardType(enum.Enum):
    """Reward type.

    Attributes
    ----------
    BINARY:
        The reward type is binary.
    CONTINUOUS:
        The reward type is continuous.
    """

    BINARY = "binary"
    CONTINUOUS = "continuous"

    def __repr__(self) -> str:

        return str(self)
