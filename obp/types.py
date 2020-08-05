# Copyright (c) ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Types."""
from typing import Union, Dict
import numpy as np

from .policy import BaseContextFreePolicy, BaseContextualPolicy

# dataset
BanditFeedback = Dict[str, Union[str, np.ndarray]]

# policy
BanditPolicy = Union[BaseContextFreePolicy, BaseContextualPolicy]
