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

    CONTEXT_FREE = enum.auto()
    CONTEXTUAL = enum.auto()
    OFFLINE = enum.auto()

    def __repr__(self) -> str:

        return str(self)
