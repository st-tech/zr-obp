from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from obp.dataset import OpenBanditDataset


@dataclass
class OBDWithInteractionFeatures(OpenBanditDataset):
    context_set: str = "1"

    def pre_process(self) -> None:

        if self.context_set == "1":
            super().pre_process()
        elif self.context_set == "2":
            self._pre_process_context_set_2()

    def _pre_process_context_set_1(self) -> None:
        """Create Context Set 1 (c.f., Section 5.2)"""

        user_cols = self.data.columns.str.contains("user_feature")
        self.context = pd.get_dummies(
            self.data.loc[:, user_cols], drop_first=True
        ).values

    def _pre_process_context_set_2(self) -> None:
        """Create Context Set 2 (c.f., Section 5.2)"""

        super().pre_process()
        affinity_cols = self.data.columns.str.contains("affinity")
        Xaffinity = self.data.loc[:, affinity_cols].values
        self.context = PCA(n_components=30).fit_transform(
            np.c_[self.context, Xaffinity]
        )
