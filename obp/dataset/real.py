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
    ----------
    behavior_policy: str
        Name of the behavior policy that generated the log data.
        Must be 'random' or 'bts'.

    campaign: str
        One of the three possible campaigns, "all", "men", and "women".

    data_path: Path, default: Path('./obd')
        Path that stores Open Bandit Dataset.

    dataset_name: str, default: 'obd'
        Name of the dataset.

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
        cls, behavior_policy: str, campaign: str, data_path: Path = Path("./obd")
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

        Returns
        ---------
        on_policy_policy_value_estimate: float
            Estimated on-policy policy value of behavior policy, i.e., :math:`T^{-1} \\sum_{t=1}^T Y_t`.
            This parameter is used as a ground-truth policy value in the evaluation of OPE estimators.

        """
        return cls(
            behavior_policy=behavior_policy, campaign=campaign, data_path=data_path
        ).reward.mean()

    def load_raw_data(self) -> None:
        """Load raw open bandit dataset."""
        self.data = pd.read_csv(self.data_path / self.raw_data_file, index_col=0)
        self.data.sort_values("timestamp", inplace=True)
        self.action = self.data["item_id"].values
        self.position = (rankdata(self.data["position"].values, "dense") - 1).astype(
            int
        )
        self.reward = self.data["click"].values
        self.pscore = self.data["propensity_score"].values

    def pre_process(self) -> None:
        """Preprocess raw open bandit dataset."""
        user_cols = self.data.columns.str.contains("user_feature")
        self.context = pd.get_dummies(
            self.data.loc[:, user_cols], drop_first=True
        ).values
        item_context = pd.read_csv(self.data_path / "item_context.csv", index_col=0)
        item_feature_0 = item_context["item_feature_0"]
        item_feature_cat = item_context.drop("item_feature_0", 1).apply(
            LabelEncoder().fit_transform
        )
        self.action_context = pd.concat([item_feature_cat, item_feature_0], 1).values

    def obtain_batch_bandit_feedback(self) -> BanditFeedback:
        """Obtain batch logged bandit feedback."""
        return dict(
            n_rounds=self.n_rounds,
            n_actions=self.n_actions,
            action=self.action,
            position=self.position,
            reward=self.reward,
            pscore=self.pscore,
            context=self.context,
            action_context=self.action_context,
        )

    def sample_bootstrap_bandit_feedback(
        self, random_state: Optional[int] = None
    ) -> BanditFeedback:
        """Sample bootstrap logged bandit feedback.

        Parameters
        -----------
        random_state: int, default: None
            Controls the random seed in sampling logged bandit dataset.

        Returns
        --------
        bootstrap_bandit_feedback: BanditFeedback
            Bootstrapped logged bandit feedback independently sampled from the original data with replacement.

        """
        random_ = check_random_state(random_state)
        bootstrap_idx = random_.choice(
            np.arange(self.n_rounds), size=self.n_rounds, replace=True
        )

        return dict(
            n_rounds=self.n_rounds,
            n_actions=self.n_actions,
            action=self.action[bootstrap_idx],
            position=self.position[bootstrap_idx],
            reward=self.reward[bootstrap_idx],
            pscore=self.pscore[bootstrap_idx],
            context=self.context[bootstrap_idx, :],
            action_context=self.action_context,
        )
