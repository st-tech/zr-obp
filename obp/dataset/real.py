# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Dataset Class for Real-World Logged Bandit Feedback."""
from dataclasses import dataclass
from logging import getLogger, basicConfig, INFO
from pathlib import Path
from typing import Optional, Union, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state, check_scalar

from .base import BaseRealBanditDataset
from ..types import BanditFeedback

logger = getLogger(__name__)
basicConfig(level=INFO)


@dataclass
class OpenBanditDataset(BaseRealBanditDataset):
    """Class for loading and preprocessing Open Bandit Dataset.

    Note
    -----
    Users are free to implement their own feature engineering by overriding the `pre_process` method.

    Parameters
    -----------
    behavior_policy: str
        Name of the behavior policy that generated the logged bandit feedback data.
        Must be either 'random' or 'bts'.

    campaign: str
        One of the three possible campaigns considered in ZOZOTOWN.
        Must be one of "all", "men", or "women".

    data_path: str or Path, default=None
        Path where the Open Bandit Dataset is stored.
        When `None` is given, this class downloads the example small-sized data.

    dataset_name: str, default='obd'
        Name of the dataset.

    References
    ------------
    Yuta Saito, Shunsuke Aihara, Megumi Matsutani, Yusuke Narita.
    "Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation.", 2020.

    """

    behavior_policy: str
    campaign: str
    data_path: Optional[Union[str, Path]] = None
    dataset_name: str = "obd"

    def __post_init__(self) -> None:
        """Initialize Open Bandit Dataset Class."""
        if self.behavior_policy not in [
            "bts",
            "random",
        ]:
            raise ValueError(
                f"behavior_policy must be either of 'bts' or 'random', but {self.behavior_policy} is given"
            )

        if self.campaign not in [
            "all",
            "men",
            "women",
        ]:
            raise ValueError(
                f"campaign must be one of 'all', 'men', or 'women', but {self.campaign} is given"
            )

        if self.data_path is None:
            logger.info(
                "When `data_path` is not given, this class downloads the example small-sized version of the Open Bandit Dataset."
            )
            self.data_path = Path(__file__).parent / "obd"
        else:
            if isinstance(self.data_path, Path):
                pass
            elif isinstance(self.data_path, str):
                self.data_path = Path(self.data_path)
            else:
                raise ValueError("data_path must be a string or Path")
        self.data_path = self.data_path / self.behavior_policy / self.campaign
        self.raw_data_file = f"{self.campaign}.csv"

        self.load_raw_data()
        self.pre_process()

    @property
    def n_rounds(self) -> int:
        """Total number of rounds contained in the logged bandit dataset."""
        return self.data.shape[0]

    @property
    def n_actions(self) -> int:
        """Number of actions."""
        return int(self.action.max() + 1)

    @property
    def dim_context(self) -> int:
        """Dimensions of context vectors."""
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
        data_path: Optional[Path] = None,
        test_size: float = 0.3,
        is_timeseries_split: bool = False,
    ) -> float:
        """Calculate on-policy policy value estimate (used as a ground-truth policy value).

        Parameters
        ----------
        behavior_policy: str
            Name of the behavior policy that generated the log data.
            Must be either 'random' or 'bts'.

        campaign: str
            One of the three possible campaigns considered in ZOZOTOWN (i.e., "all", "men", and "women").

        data_path: Path, default=None
            Path where the Open Bandit Dataset exists.
            When `None` is given, this class downloads the example small-sized version of the dataset.

        test_size: float, default=0.3
            Proportion of the dataset included in the test split.
            If float, should be between 0.0 and 1.0.
            This argument matters only when `is_timeseries_split=True` (the out-sample case).

        is_timeseries_split: bool, default=False
            If true, split the original logged bandit feedback data by time series.

        Returns
        ---------
        on_policy_policy_value_estimate: float
            Policy value of the behavior policy estimated by on-policy estimation, i.e., :math:`\\mathbb{E}_{\\mathcal{D}} [r_t]`.
            where :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`T` observations in :math:`\\mathcal{D}`.
            This parameter is used as a ground-truth policy value in the evaluation of OPE estimators.

        """
        bandit_feedback = cls(
            behavior_policy=behavior_policy, campaign=campaign, data_path=data_path
        ).obtain_batch_bandit_feedback(
            test_size=test_size, is_timeseries_split=is_timeseries_split
        )
        if is_timeseries_split:
            bandit_feedback_test = bandit_feedback[1]
        else:
            bandit_feedback_test = bandit_feedback
        return bandit_feedback_test["reward"].mean()

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
        This is the default feature engineering and please override this method to
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
    ) -> Union[BanditFeedback, Tuple[BanditFeedback, BanditFeedback]]:
        """Obtain batch logged bandit feedback.

        Parameters
        -----------
        test_size: float, default=0.3
            Proportion of the dataset included in the test split.
            If float, should be between 0.0 and 1.0.
            This argument matters only when `is_timeseries_split=True` (the out-sample case).

        is_timeseries_split: bool, default=False
            If true, split the original logged bandit feedback data into train and test sets based on time series.

        Returns
        --------
        bandit_feedback: BanditFeedback
            A dictionary containing batch logged bandit feedback data collected by a behavior policy.
            The keys of the dictionary are as follows.
            - n_rounds: number of rounds (size) of the logged bandit data
            - n_actions: number of actions (:math:`|\mathcal{A}|`)
            - action: action variables sampled by a behavior policy
            - position: positions where actions are recommended
            - reward: reward variables
            - pscore: action choice probabilities by a behavior policy
            - context: context vectors such as user-related features and user-item affinity scores
            - action_context: item-related context vectors

        """
        if not isinstance(is_timeseries_split, bool):
            raise TypeError(
                f"`is_timeseries_split` must be a bool, but {type(is_timeseries_split)} is given"
            )

        if is_timeseries_split:
            check_scalar(
                test_size,
                name="target_size",
                target_type=(float),
                min_val=0.0,
                max_val=1.0,
            )
            n_rounds_train = np.int(self.n_rounds * (1.0 - test_size))
            bandit_feedback_train = dict(
                n_rounds=n_rounds_train,
                n_actions=self.n_actions,
                action=self.action[:n_rounds_train],
                position=self.position[:n_rounds_train],
                reward=self.reward[:n_rounds_train],
                pscore=self.pscore[:n_rounds_train],
                context=self.context[:n_rounds_train],
                action_context=self.action_context,
            )
            bandit_feedback_test = dict(
                n_rounds=(self.n_rounds - n_rounds_train),
                n_actions=self.n_actions,
                action=self.action[n_rounds_train:],
                position=self.position[n_rounds_train:],
                reward=self.reward[n_rounds_train:],
                pscore=self.pscore[n_rounds_train:],
                context=self.context[n_rounds_train:],
                action_context=self.action_context,
            )
            return bandit_feedback_train, bandit_feedback_test
        else:
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
        self,
        sample_size: Optional[int] = None,
        test_size: float = 0.3,
        is_timeseries_split: bool = False,
        random_state: Optional[int] = None,
    ) -> BanditFeedback:
        """Obtain bootstrap logged bandit feedback.

        Parameters
        -----------
        sample_size: int, default=None
            Number of data sampled by bootstrap.
            When None is given, the original data size (n_rounds) is used as `sample_size`.
            The value must be smaller than the original data size.

        test_size: float, default=0.3
            Proportion of the dataset included in the test split.
            If float, should be between 0.0 and 1.0.
            This argument matters only when `is_timeseries_split=True` (the out-sample case).

        is_timeseries_split: bool, default=False
            If true, split the original logged bandit feedback data into train and test sets based on time series.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        --------
        bandit_feedback: BanditFeedback
            A dictionary containing logged bandit feedback data sampled independently from the original data with replacement.
            The keys of the dictionary are as follows.
            - n_rounds: number of rounds (size) of the logged bandit data
            - n_actions: number of actions
            - action: action variables sampled by a behavior policy
            - position: positions where actions are recommended by a behavior policy
            - reward: reward variables
            - pscore: action choice probabilities by a behavior policy
            - context: context vectors such as user-related features and user-item affinity scores
            - action_context: item-related context vectors

        """
        if is_timeseries_split:
            bandit_feedback = self.obtain_batch_bandit_feedback(
                test_size=test_size, is_timeseries_split=is_timeseries_split
            )[0]
        else:
            bandit_feedback = self.obtain_batch_bandit_feedback(
                test_size=test_size, is_timeseries_split=is_timeseries_split
            )
        n_rounds = bandit_feedback["n_rounds"]
        if sample_size is None:
            sample_size = bandit_feedback["n_rounds"]
        else:
            check_scalar(
                sample_size,
                name="sample_size",
                target_type=(int),
                min_val=0,
                max_val=n_rounds,
            )
        random_ = check_random_state(random_state)
        bootstrap_idx = random_.choice(
            np.arange(n_rounds), size=sample_size, replace=True
        )
        for key_ in ["action", "position", "reward", "pscore", "context"]:
            bandit_feedback[key_] = bandit_feedback[key_][bootstrap_idx]
        bandit_feedback["n_rounds"] = sample_size
        return bandit_feedback
