# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Abstract Base Class for Logged Bandit Feedback."""
from abc import ABCMeta, abstractmethod


class BaseBanditDataset(metaclass=ABCMeta):
    """Base Class for Synthetic Bandit Dataset."""

    @abstractmethod
    def obtain_batch_bandit_feedback(self) -> None:
        """Obtain batch logged bandit feedback."""
        raise NotImplementedError


class BaseRealBanditDataset(BaseBanditDataset):
    """Base Class for Real-World Bandit Dataset."""

    @abstractmethod
    def load_raw_data(self) -> None:
        """Load raw dataset."""
        raise NotImplementedError

    @abstractmethod
    def pre_process(self) -> None:
        """Preprocess raw dataset."""
        raise NotImplementedError
