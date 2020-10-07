# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Dataset Class for Real-World Logged Bandit Feedback."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
from .base import BaseRealBanditDataset
from ..types import BanditFeedback


@dataclass
class OpenBanditDataset(BaseRealBanditDataset):
    """Class for loading and preprocessing Open Bandit Dataset.

    Note
    -----
    Users are free to implement their own feature engineering by overriding `pre_process` method.

    Parameters
    -----------
    behavior_policy: str
        Name of the behavior policy that generated the log data.
        Must be 'random' or 'bts'.

    campaign: str
        One of the three possible campaigns, "all", "men", and "women".

    data_path: Path, default: Path('./obd')
        Path that stores Open Bandit Dataset.

    dataset_name: str, default: 'obd'
        Name of the dataset.

    References
    ------------
    Yuta Saito, Shunsuke Aihara, Megumi Matsutani, Yusuke Narita.
    "A Large-scale Open Dataset for Bandit Algorithms.", 2020.

    """

    behavior_policy: str
    campaign: str
    data_path: Path = Path("./obd")
    dataset_name: str = "obd"

    def __post_init__(self) -> None:
        """Initialize Open Bandit Dataset Class."""
        assert self.behavior_policy in [
            "bts",
            "random",
        ], f"behavior_policy must be either of 'bts' or 'random', but {self.behavior_policy} is given"
        assert self.campaign in [
            "all",
            "men",
            "women",
        ], f"campaign must be one of 'all', 'men', and 'women', but {self.campaign} is given"
        assert isinstance(self.data_path, Path), f"data_path must be a Path"

        self.data_path = self.data_path / self.behavior_policy / self.campaign
        self.raw_data_file = f"{self.campaign}.csv"

        self.load_raw_data()
        self.pre_process()

    @property
    def n_rounds(self) -> int:
        """Total number of rounds in the dataset."""
        return self.data.shape[0]

    @property
    def n_actions(self) -> int:
        """Number of actions."""
        return int(self.action.max() + 1)

    @property
    def dim_context(self) -> int:
        """Number of dimensions of context vectors."""
        return self.context.shape[1]

    @property
    def len_list(self) -> int:
        """Length of recommendation lists."""
        return int(self.position.max() + 1)

    @classmethod
    def calc_on_policy_policy_value_estimate(
        cls,
        behavior_policy: str,
        campaign: str,
        data_path: Path = Path("./obd"),
        test_size: float = 0.3,
        is_timeseries_split: bool = False,
    ) -> float:
        """Calculate on-policy policy value estimate (used as a ground-truth policy value).

        Parameters
        ----------
        behavior_policy: str
            Name of the behavior policy that generated the log data.
            Must be 'random' or 'bts'.

        campaign: str
            One of the three possible campaigns (i.e., "all", "men", and "women").

        data_path: Path, default: Path('./obd')
            Path that stores Open Bandit Dataset.

        test_size: float, default=0.3
            If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.

        is_timeseries_split: bool, default: False
            If true, split the original logged badnit feedback data by time series.

        Returns
        ---------
        on_policy_policy_value_estimate: float
            Estimated on-policy policy value of behavior policy, i.e., :math:`T^{-1} \\sum_{t=1}^T Y_t`.
            This parameter is used as a ground-truth policy value in the evaluation of OPE estimators.

        """
        return (
            cls(behavior_policy=behavior_policy, campaign=campaign, data_path=data_path)
            .obtain_batch_bandit_feedback(
                test_size=test_size, is_timeseries_split=is_timeseries_split
            )["reward_test"]
            .mean()
        )

    def load_raw_data(self) -> None:
        """Load raw open bandit dataset."""
        self.data = pd.read_csv(self.data_path / self.raw_data_file, index_col=0)
        self.item_context = pd.read_csv(
            self.data_path / "item_context.csv", index_col=0
        )
        self.data.sort_values("timestamp", inplace=True)
        self.action = self.data["item_id"].values
        self.position = (rankdata(self.data["position"].values, "dense") - 1).astype(
            int
        )
        self.reward = self.data["click"].values
        self.pscore = self.data["propensity_score"].values

    def pre_process(self) -> None:
        """Preprocess raw open bandit dataset.

        Note
        -----
        This is the default feature engineering and please overide this method to
        implement your own preprocessing.
        see https://github.com/st-tech/zr-obp/blob/master/examples/examples_with_obd/custom_dataset.py for example.

        """
        user_cols = self.data.columns.str.contains("user_feature")
        self.context = pd.get_dummies(
            self.data.loc[:, user_cols], drop_first=True
        ).values
        item_feature_0 = self.item_context["item_feature_0"]
        item_feature_cat = self.item_context.drop("item_feature_0", 1).apply(
            LabelEncoder().fit_transform
        )
        self.action_context = pd.concat([item_feature_cat, item_feature_0], 1).values

    def obtain_batch_bandit_feedback(
        self, test_size: float = 0.3, is_timeseries_split: bool = False
    ) -> BanditFeedback:
        """Obtain batch logged bandit feedback.

        Parameters
        -----------
        test_size: float, default=0.3
            If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.

        is_timeseries_split: bool, default: False
            If true, split the original logged badnit feedback data by time series.

        Returns
        --------
        bandit_feedback: BanditFeedback
            Logged bandit feedback collected by the behavior policy.

        """
        if is_timeseries_split:
            assert isinstance(test_size, float) & (
                0 < test_size < 1
            ), f"test_size must be a float between 0 and 1, but {test_size} is given"
            n_rounds_train = np.int(self.n_rounds * (1.0 - test_size))
            return dict(
                n_rounds=n_rounds_train,
                n_actions=self.n_actions,
                action=self.action[:n_rounds_train],
                action_test=self.action[n_rounds_train:],
                position=self.position[:n_rounds_train],
                position_test=self.position[n_rounds_train:],
                reward=self.reward[:n_rounds_train],
                reward_test=self.reward[n_rounds_train:],
                pscore=self.pscore[:n_rounds_train],
                pscore_test=self.pscore[n_rounds_train:],
                context=self.context[:n_rounds_train],
                context_test=self.context[n_rounds_train:],
                action_context=self.action_context,
            )
        else:
            return dict(
                n_rounds=self.n_rounds,
                n_actions=self.n_actions,
                action=self.action,
                position=self.position,
                reward=self.reward,
                reward_test=self.reward,
                pscore=self.pscore,
                context=self.context,
                action_context=self.action_context,
            )

    def sample_bootstrap_bandit_feedback(
        self,
        test_size: float = 0.3,
        is_timeseries_split: bool = False,
        random_state: Optional[int] = None,
    ) -> BanditFeedback:
        """Sample bootstrap logged bandit feedback.

        Parameters
        -----------
        test_size: float, default=0.3
            If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.

        is_timeseries_split: bool, default: False
            If true, split the original logged badnit feedback data by time series.

        random_state: int, default: None
            Controls the random seed in sampling logged bandit dataset.

        Returns
        --------
        bootstrap_bandit_feedback: BanditFeedback
            Bootstrapped logged bandit feedback independently sampled from the original data with replacement.

        """
        bandit_feedback = self.obtain_batch_bandit_feedback(
            test_size=test_size, is_timeseries_split=is_timeseries_split
        )
        n_rounds = bandit_feedback["n_rounds"]
        random_ = check_random_state(random_state)
        bootstrap_idx = random_.choice(np.arange(n_rounds), size=n_rounds, replace=True)
        for key_ in ["action", "position", "reward", "pscore", "context"]:
            bandit_feedback[key_] = bandit_feedback[key_][bootstrap_idx]
        return bandit_feedback
