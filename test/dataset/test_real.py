import pytest
import numpy as np
import pandas as pd

from obp.dataset import OpenBanditDataset


def test_real_init():
    # behavior_policy
    with pytest.raises(ValueError):
        OpenBanditDataset(behavior_policy="aaa", campaign="all")

    # campaign
    with pytest.raises(ValueError):
        OpenBanditDataset(behavior_policy="random", campaign="aaa")

    # data_path
    with pytest.raises(ValueError):
        OpenBanditDataset(behavior_policy="random", campaign="all", data_path="raw_str_path")

    # load_raw_data
    opd = OpenBanditDataset(behavior_policy="random", campaign="all")
    # check the value exists and has the right type
    assert (
            isinstance(opd.data, pd.DataFrame)
            and isinstance(opd.item_context, pd.DataFrame)
            and isinstance(opd.action, np.ndarray)
            and isinstance(opd.position, np.ndarray)
            and isinstance(opd.reward, np.ndarray)
            and isinstance(opd.pscore, np.ndarray)
    )

    # pre_process (context and action_context)
    assert (
        isinstance(opd.context, np.ndarray)
        and isinstance(opd.action_context, np.ndarray)
    )

