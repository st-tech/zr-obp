# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Types."""
from typing import Dict
from typing import Union

import numpy as np

from .policy import BaseContextFreePolicy
from .policy import BaseContextualPolicy


# dataset
BanditFeedback = Dict[str, Union[int, np.ndarray]]

# policy
BanditPolicy = Union[BaseContextFreePolicy, BaseContextualPolicy]
