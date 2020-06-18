from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Union

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

TRAIN_DICT = Dict[str, Union[str, np.ndarray]]


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
    Users are free to implement their own featuer engineering by overriding `pre_process` method.

    Parameters
    ----------
    behavior policy: str
        Name of the behavior policy that generated the log data.

    campaign: str
        One of the three possible campaigns (i.e., "all", "men's", and "women's").

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
        self.raw_data_file = f'{self.campaign}.csv'

        self.load_raw_data()
        self.pre_process()
        self.pre_process_for_regression_model()

        self.n_data = self.action.shape[0]
        self.n_actions = self.action.max() + 1
        self.dim_context = self.X_policy.shape[1]

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
        self.X_policy = pd.get_dummies(self.data.loc[:, user_cols], drop_first=True).values

    def pre_process_for_regression_model(self) -> None:
        """Preprocess raw open bandit dataset for training a regression model."""
        user_cols = self.data.columns.str.contains('user_feature')
        self.X_user = self.data.loc[:, user_cols].apply(LabelEncoder().fit_transform).values
        item_context = pd.read_csv(self.data_path / 'item_context.csv', index_col=0)
        item_feature_0 = item_context['item_feature_0']
        item_feature_cat = item_context.drop('item_feature_0', 1).apply(LabelEncoder().fit_transform)
        self.X_action = pd.concat([item_feature_cat, item_feature_0], 1).values
        self.X_reg = np.c_[self.pos, self.X_user, self.X_action[self.action, :]]

    def split_data(self,
                   test_size: Union[int, float] = 0.3,
                   is_timeseries_split: bool = False,
                   random_state: int = 0) -> Tuple[TRAIN_DICT, TRAIN_DICT]:
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
        train: TRAIN_DICT
            Dictionary storing the training set after preprocessing.

        test: TRAIN_DICT
            Dictionary storing the test set after preprocessing.
        """
        if is_timeseries_split:
            test_size = test_size if isinstance(test_size, int) else np.int(test_size * self.n_data)
            train_size = np.int(self.n_data - test_size)
            action_train, action_test = self.action[:train_size], self.action[train_size:]
            pos_train, pos_test = self.pos[:train_size], self.pos[train_size:]
            reward_train, reward_test = self.reward[:train_size], self.reward[train_size:]
            pscore_train, pscore_test = self.pscore[:train_size], self.pscore[train_size:]
            X_policy_train, X_policy_test = self.X_policy[:train_size], self.X_policy[train_size:]
            X_reg_train, X_reg_test = self.X_reg[:train_size], self.X_reg[train_size:]
            X_user_train, X_user_test = self.X_user[:train_size], self.X_user[train_size:]
        else:
            action_train, action_test, pos_train, pos_test, reward_train, reward_test, pscore_train, pscore_test,\
                X_policy_train, X_policy_test, X_reg_train, X_reg_test, X_user_train, X_user_test =\
                train_test_split(
                    self.action,
                    self.pos,
                    self.reward,
                    self.pscore,
                    self.X_policy,
                    self.X_reg,
                    self.X_user,
                    test_size=test_size,
                    random_state=random_state)

        self.train_size = action_train.shape[0]
        self.test_size = action_test.shape[0]
        train = dict(
            n_data=self.train_size,
            n_actions=self.n_actions,
            action=action_train,
            position=pos_train,
            reward=reward_train,
            pscore=pscore_train,
            X_policy=X_policy_train,
            X_reg=X_reg_train,
            X_user=X_user_train)
        test = dict(
            n_data=self.train_size,
            n_actions=self.n_actions,
            action=action_test,
            position=pos_test,
            reward=reward_test,
            pscore=pscore_test,
            X_policy=X_policy_test,
            X_reg=X_reg_test,
            X_user=X_user_test)

        return train, test
