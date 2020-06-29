# Copyright (c) ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Dataset Class for Logged Bandit Feedback."""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Union

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

LogBanditFeedback = Dict[str, Union[str, np.ndarray]]


class BaseBanditDataset(metaclass=ABCMeta):
    """Base Bandit Dataset Class."""

    @abstractmethod
    def load_raw_data(self) -> None:
        """Load raw dataset."""
        raise NotImplementedError()

    @abstractmethod
    def pre_process(self) -> None:
        """Preprocess raw dataset."""
        raise NotImplementedError()

    @abstractmethod
    def split_data(self) -> None:
        """Split dataset into two folds."""
        raise NotImplementedError()


@dataclass
class OpenBanditDataset(BaseBanditDataset):
    """A class for loading and preprocessing Open Bandit Dataset.

    Note
    -----
    Users are free to implement their own featuer engineering by overriding `pre_process` method.

    Parameters
    ----------
    behavior_policy: str
        Name of the behavior policy that generated the log data.
        Must be 'random' or 'bts'.

    campaign: str
        One of the three possible campaigns (i.e., "all", "men", and "women").

    data_path: Path, default: Path('./obd')
        Path that stores Open Bandit Dataset.

    dataset_name: str, default: 'obd'
        Name of the dataset.

    """
    behavior_policy: str
    campaign: str
    data_path: Path = Path('./obd')
    dataset_name: str = 'obd'

    def __post_init__(self) -> None:
        """Initialize Open Bandit Dataset Class."""
        self.data_path = self.data_path / self.behavior_policy / self.campaign
        self.raw_data_file = f'{self.campaign}.zip'

        self.load_raw_data()
        self.pre_process()

        self.n_rounds = self.action.shape[0]
        self.n_actions = self.action.max() + 1
        self.dim_context = self.context.shape[1]

    def load_raw_data(self) -> None:
        """Load raw open bandit dataset."""

        self.data = pd.read_csv(self.data_path / self.raw_data_file, index_col=0)
        self.data.sort_values('timestamp', inplace=True)
        self.action = self.data['item_id'].values
        self.pos = (rankdata(self.data['position'].values, 'dense') - 1).astype(int)
        self.len_list = self.pos.max() + 1
        self.reward = self.data['click'].values
        self.pscore = self.data['propensity_score'].values

    def pre_process(self) -> None:
        """Preprocess raw open bandit dataset."""
        user_cols = self.data.columns.str.contains('user_feature')
        self.context = pd.get_dummies(self.data.loc[:, user_cols], drop_first=True).values
        item_context = pd.read_csv(self.data_path / 'item_context.csv', index_col=0)
        item_feature_0 = item_context['item_feature_0']
        item_feature_cat = item_context.drop('item_feature_0', 1).apply(LabelEncoder().fit_transform)
        self.action_context = pd.concat([item_feature_cat, item_feature_0], 1).values

    def split_data(self,
                   test_size: Union[int, float] = 0.3,
                   is_timeseries_split: bool = False,
                   random_state: int = 0) -> Tuple[LogBanditFeedback, LogBanditFeedback]:
        """Split dataset into training and test sets.

        Parameters
        ----------
        test_size: int, float, default=0.3
            If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
            If int, represents the absolute number of test samples.

        is_timeseries_split: bool, default: False
            If true, split data by time series.

        random_state: int, default: 0
            Controls the shuffling applied to the data before applying the split.

        Returns:
        ----------
        train: LogBanditFeedback
            Dictionary storing the training set after preprocessing.

        test: LogBanditFeedback
            Dictionary storing the test set after preprocessing.
        """
        if is_timeseries_split:
            test_size = test_size if isinstance(test_size, int) else np.int(test_size * self.n_rounds)
            train_size = np.int(self.n_rounds - test_size)
            action_train, action_test = self.action[:train_size], self.action[train_size:]
            pos_train, pos_test = self.pos[:train_size], self.pos[train_size:]
            reward_train, reward_test = self.reward[:train_size], self.reward[train_size:]
            pscore_train, pscore_test = self.pscore[:train_size], self.pscore[train_size:]
            context_train, context_test = self.context[:train_size], self.context[train_size:]
        else:
            action_train, action_test, pos_train, pos_test, reward_train, reward_test,\
                pscore_train, pscore_test, context_train, context_test =\
                train_test_split(
                    self.action,
                    self.pos,
                    self.reward,
                    self.pscore,
                    self.context,
                    test_size=test_size,
                    random_state=random_state)

        self.train_size = action_train.shape[0]
        self.test_size = action_test.shape[0]
        train = dict(
            n_rounds=self.train_size,
            n_actions=self.n_actions,
            action=action_train,
            position=pos_train,
            reward=reward_train,
            pscore=pscore_train,
            context=context_train,
            action_context=self.action_context)
        test = dict(
            n_rounds=self.test_size,
            n_actions=self.n_actions,
            action=action_test,
            position=pos_test,
            reward=reward_test,
            pscore=pscore_test,
            context=context_test,
            action_context=self.action_context)

        return train, test
