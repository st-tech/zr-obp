import enum


class PolicyType(enum.Enum):
    """Policy type.

    Attributes
    ----------
    CONTEXT_FREE:
        The policy type is contextfree.
    CONTEXTUAL:
        The policy type is contextual.
    OFFLINE:
        The policy type is offline.
    """

    CONTEXT_FREE = 0
    CONTEXTUAL = 1
    OFFLINE = 2

    def __repr__(self) -> str:

        return str(self)
