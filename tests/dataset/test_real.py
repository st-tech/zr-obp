import pytest
import numpy as np
import pandas as pd

from typing import Tuple

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
        OpenBanditDataset(
            behavior_policy="random", campaign="all", data_path="raw_str_path"
        )

    # load_raw_data
    obd = OpenBanditDataset(behavior_policy="random", campaign="all")
    # check the value exists and has the right type
    assert (
        isinstance(obd.data, pd.DataFrame)
        and isinstance(obd.item_context, pd.DataFrame)
        and isinstance(obd.action, np.ndarray)
        and isinstance(obd.position, np.ndarray)
        and isinstance(obd.reward, np.ndarray)
        and isinstance(obd.pscore, np.ndarray)
    )

    # pre_process (context and action_context)
    assert isinstance(obd.context, np.ndarray) and isinstance(
        obd.action_context, np.ndarray
    )


def test_obtain_batch_bandit_feedback():
    # invalid test_size
    with pytest.raises(ValueError):
        dataset = OpenBanditDataset(behavior_policy="random", campaign="all")
        dataset.obtain_batch_bandit_feedback(is_timeseries_split=True, test_size=1.3)

    with pytest.raises(ValueError):
        dataset = OpenBanditDataset(behavior_policy="random", campaign="all")
        dataset.obtain_batch_bandit_feedback(is_timeseries_split=True, test_size=-0.5)

    # existence of keys
    # is_timeseries_split=False (default)
    dataset = OpenBanditDataset(behavior_policy="random", campaign="all")
    bandit_feedback = dataset.obtain_batch_bandit_feedback()

    assert "n_rounds" in bandit_feedback.keys()
    assert "n_actions" in bandit_feedback.keys()
    assert "action" in bandit_feedback.keys()
    assert "position" in bandit_feedback.keys()
    assert "reward" in bandit_feedback.keys()
    assert "pscore" in bandit_feedback.keys()
    assert "context" in bandit_feedback.keys()
    assert "action_context" in bandit_feedback.keys()

    # is_timeseries_split=True
    bandit_feedback_timeseries = dataset.obtain_batch_bandit_feedback(
        is_timeseries_split=True
    )
    assert isinstance(bandit_feedback_timeseries, Tuple)
    bandit_feedback_train = bandit_feedback_timeseries[0]
    bandit_feedback_test = bandit_feedback_timeseries[1]

    bf_elems = {
        "n_rounds",
        "n_actions",
        "action",
        "position",
        "reward",
        "pscore",
        "context",
        "action_context",
    }
    assert all(k in bandit_feedback_train.keys() for k in bf_elems)
    assert all(k in bandit_feedback_test.keys() for k in bf_elems)


def test_calc_on_policy_policy_value_estimate():
    ground_truth_policy_value = OpenBanditDataset.calc_on_policy_policy_value_estimate(
        behavior_policy="random", campaign="all"
    )
    assert isinstance(ground_truth_policy_value, float)


def test_sample_bootstrap_bandit_feedback():
    dataset = OpenBanditDataset(behavior_policy="random", campaign="all")
    bandit_feedback = dataset.obtain_batch_bandit_feedback()
    bootstrap_bf = dataset.sample_bootstrap_bandit_feedback()

    assert len(bandit_feedback["action"]) == len(bootstrap_bf["action"])
    assert len(bandit_feedback["position"]) == len(bootstrap_bf["position"])
    assert len(bandit_feedback["reward"]) == len(bootstrap_bf["reward"])
    assert len(bandit_feedback["pscore"]) == len(bootstrap_bf["pscore"])
    assert len(bandit_feedback["context"]) == len(bootstrap_bf["context"])
