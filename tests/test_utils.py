import numpy as np

from obp.utils import sample_action_fast
from obp.utils import softmax


def test_sample_action_fast():
    n_rounds = 10
    n_actions = 5
    n_sim = 100000

    true_probs = softmax(np.random.normal(size=(n_rounds, n_actions)))
    sampled_action_list = list()
    for _ in np.arange(n_sim):
        sampled_action_list.append(sample_action_fast(true_probs)[:, np.newaxis])

    sampled_action_arr = np.concatenate(sampled_action_list, 1)
    for i in np.arange(n_rounds):
        sampled_action_counts = np.unique(sampled_action_arr[i], return_counts=True)[1]
        empirical_probs = sampled_action_counts / n_sim
        assert np.isclose(true_probs[i], empirical_probs, rtol=5e-2, atol=1e-3).all()
