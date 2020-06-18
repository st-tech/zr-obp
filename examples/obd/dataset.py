from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from obp.dataset import OpenBanditDataset, TRAIN_DICT


@dataclass
class OBDWithContextSets(OpenBanditDataset):
    context_set: str = '1'

    def pre_process(self) -> None:
        if self.context_set == '1':
            self._pre_process_context_set_1()
        elif self.context_set == '2':
            self._pre_process_context_set_2()

    def sample_bootstrap(self, train: TRAIN_DICT) -> TRAIN_DICT:
        """Sample bootstrap training sampels to make the bagging prediction."""
        boot_idx = np.random.choice(np.arange(self.train_size), size=self.train_size)
        train_boot = dict(
            n_data=self.train_size,
            n_actions=self.n_actions,
            action=train['action'][boot_idx],
            position=train['position'][boot_idx],
            reward=train['reward'][boot_idx],
            pscore=train['pscore'][boot_idx],
            X_policy=train['X_policy'][boot_idx, :],
            X_reg=train['X_reg'][boot_idx, :],
            X_user=train['X_user'][boot_idx, :])

        return train_boot

    def _pre_process_context_set_1(self) -> None:
        """Create Context Set 1 (c.f., Section 5.2)"""

        user_cols = self.data.columns.str.contains('user_feature')
        self.X_policy = pd.get_dummies(self.data.loc[:, user_cols], drop_first=True).values

    def _pre_process_context_set_2(self) -> None:
        """Create Context Set 2 (c.f., Section 5.2)"""

        user_cols = self.data.columns.str.contains('user_feature')
        Xuser = pd.get_dummies(self.data.loc[:, user_cols], drop_first=True).values
        affinity_cols = self.data.columns.str.contains('affinity')
        Xaffinity = self.data.loc[:, affinity_cols].values
        self.X_policy = PCA(n_components=30).fit_transform(np.c_[Xuser, Xaffinity])
