from unittest import mock

import numpy as np
import pytest

from obp.dataset.synthetic import logistic_sparse_reward_function, _base_reward_function
from obp.simulator.coefficient_drifter import CoefficientDrifter
from obp.simulator.simulator import BanditEnvironmentSimulator


def test_coefficient_tracker_can_shift_expected_rewards_with_syntethic_dataset_generator():
    drifter = CoefficientDrifter(drift_interval=3)

    dataset = BanditEnvironmentSimulator(
        n_actions=3,
        dim_context=4,
        reward_type="binary",  # "binary" or "continuous"
        reward_function=logistic_sparse_reward_function,
        coef_function=drifter.get_coefficients,
        random_state=12345,
    )

    bandit_dataset = dataset.obtain_batch_bandit_feedback(n_rounds=4)


class MockCoefSample:
    n_samples = 0

    def fake_sample(
        self, effective_dim_action_context, effective_dim_context, random_, **kwargs
    ):
        self.n_samples += 1
        context_coef_ = self.n_samples * np.ones(effective_dim_context)
        action_coef_ = self.n_samples * np.ones(effective_dim_action_context)
        context_action_coef_ = self.n_samples * np.ones(
            (effective_dim_context, effective_dim_action_context)
        )

        return context_coef_, action_coef_, context_action_coef_


def test_coefficient_tracker_can_shift_expected_rewards_instantly_based_on_configured_intervals():
    with mock.patch(
        "obp.dataset.synthetic.sample_random_uniform_coefficients",
        MockCoefSample().fake_sample,
    ):
        drifter = CoefficientDrifter(drift_interval=3)

        context = np.asarray(
            [
                [1, 2],
                [3, 2],
                [3, 2],
                [3, 2],
            ]
        )
        action_context = np.asarray(
            [
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
            ]
        )
        actual_expected_rewards = _base_reward_function(
            context,
            action_context,
            degree=5,
            effective_dim_ratio=1.0,
            coef_function=drifter.get_coefficients,
            random_state=12345,
        )

    expected_expected_rewards = np.asarray(
        [
            [-7.88993004, -7.88993004, -7.88993004],  # This round has a different context and should have diff E[r]
            [ 0.9467916 ,  0.9467916 ,  0.9467916 ],  # The next two rounds have the same context and should have identical
            [ 0.9467916 ,  0.9467916 ,  0.9467916 ],  # E[r]
            [ 5.99634683,  5.99634683,  5.99634683],  # This round has the same context but has experienced drift.
        ]
    )

    assert np.allclose(actual_expected_rewards, expected_expected_rewards)


def test_coefficient_tracker_can_shift_coefficient_instantly_based_on_configured_interval():
    with mock.patch(
        "obp.dataset.synthetic.sample_random_uniform_coefficients",
        MockCoefSample().fake_sample,
    ):
        effective_dim_context = 4
        effective_dim_action_context = 3
        drifter = CoefficientDrifter(
            drift_interval=3,
            effective_dim_context=effective_dim_context,
            effective_dim_action_context=effective_dim_action_context,
        )

        actual_context_coef, _, _ = drifter.get_coefficients(n_rounds=4)

    expected_context_coef = np.asarray(
        [
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],  # AFTER THIS ROUND, THE COEFS CHANGE ABRUPTLY
            [3, 3, 3, 3],
        ]
    )

    assert np.allclose(actual_context_coef, expected_context_coef)


def test_coefficient_tracker_can_shift_linearly_instantly_based_on_configured_transition_period():
    with mock.patch(
        "obp.dataset.synthetic.sample_random_uniform_coefficients",
        MockCoefSample().fake_sample,
    ):
        drifter = CoefficientDrifter(
            drift_interval=4,
            transition_period=2,
            effective_dim_context=2,
            effective_dim_action_context=2,
        )

        actual_context_coef, _, _ = drifter.get_coefficients(n_rounds=8)

    expected_context_coef = np.asarray(
        [
            [2.0, 2.0],
            [2.0, 2.0],  # First two rounds are the same
            [2.33333333, 2.33333333],  # Next two rounds slowly transition
            [2.66666667, 2.66666667],
            [3.0, 3.0],  # Now we start in the new coef again
            [3.0, 3.0],
            [3.33333333, 3.33333333],  # Now we start transitioning again.
            [3.66666667, 3.66666667],
        ]
    )

    assert np.allclose(actual_context_coef, expected_context_coef)


def test_coefficient_tracker_can_shift_weighted_sampled_based_on_configured_transition_period():
    with mock.patch(
        "obp.dataset.synthetic.sample_random_uniform_coefficients",
        MockCoefSample().fake_sample,
    ):
        drifter = CoefficientDrifter(
            drift_interval=4,
            transition_period=2,
            transition_type="weighted_sampled",
            effective_dim_context=2,
            effective_dim_action_context=2,
        )

        actual_context_coef, _, _ = drifter.get_coefficients(n_rounds=8)

    expected_context_coef = np.asarray(
        [
            [2.0, 2.0],
            [2.0, 2.0],  # First two rounds are the same
            [2.0, 2.0],  # Next two rounds are weighted sampled
            [3.0, 3.0],
            [3.0, 3.0],  # Now we start in the new coef again
            [3.0, 3.0],
            [4.0, 4.0],  # Next two rounds are weighted sampled again
            [4.0, 4.0],
        ]
    )

    assert np.allclose(actual_context_coef, expected_context_coef)


def test_coefficient_tracker_can_shift_instantly_back_and_forth_between_seasons_using_seasonality_flag():
    with mock.patch(
        "obp.dataset.synthetic.sample_random_uniform_coefficients",
        MockCoefSample().fake_sample,
    ):
        drifter = CoefficientDrifter(
            drift_interval=2,
            transition_period=0,
            seasonal=True,
            effective_dim_context=2,
            effective_dim_action_context=2,
        )

        actual_context_coef, _, _ = drifter.get_coefficients(n_rounds=8)

    expected_context_coef = np.asarray(
        [
            [2.0, 2.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [3.0, 3.0],
            [2.0, 2.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [3.0, 3.0],
        ]
    )

    assert np.allclose(actual_context_coef, expected_context_coef)


def test_coefficient_tracker_can_shift_instantly_under_base_coeficient():
    with mock.patch(
        "obp.dataset.synthetic.sample_random_uniform_coefficients",
        MockCoefSample().fake_sample,
    ):
        drifter = CoefficientDrifter(
            drift_interval=2,
            transition_period=0,
            seasonal=True,
            base_coefficient_weight=0.8,
            effective_dim_context=2,
            effective_dim_action_context=2,
        )

        actual_context_coef, _, _ = drifter.get_coefficients(n_rounds=8)

        expected_context_coef = np.asarray(
            [
                [1.2, 1.2],
                [1.2, 1.2],
                [1.4, 1.4],
                [1.4, 1.4],
                [1.2, 1.2],
                [1.2, 1.2],
                [1.4, 1.4],
                [1.4, 1.4],
            ]
        )

    assert np.allclose(actual_context_coef, expected_context_coef)


def test_coefficient_tracker_update_coef_makes_next_coef_current_coef_and_samples_new_next_coef():
    with mock.patch(
        "obp.dataset.synthetic.sample_random_uniform_coefficients",
        MockCoefSample().fake_sample,
    ):
        drifter = CoefficientDrifter(
            drift_interval=4, effective_dim_context=1, effective_dim_action_context=1
        )

        drifter.context_coefs[0] = [1]
        drifter.context_coefs[1] = [2]

        drifter.update_coef()

        assert drifter.context_coefs[0] == [2]
        assert np.allclose(drifter.context_coefs[1], [4.0])


def test_coefficient_tracker_update_coef_samples_both_new_curr_and_next_on_first_pull():
    drifter = CoefficientDrifter(
        drift_interval=4,
    )

    assert len(drifter.context_coefs) == 0

    drifter.effective_dim_context = 1
    drifter.effective_dim_action_context = 1

    drifter.update_coef()

    assert drifter.context_coefs[0] is not None
    assert drifter.context_coefs[1] is not None
    assert not np.allclose(drifter.context_coefs[0], drifter.context_coefs[1])


def test_coefficient_tracker_can_set_effective_dim_context_on_first_sample():
    effective_dim_context = 4
    effective_dim_action_context = 3
    drifter = CoefficientDrifter(drift_interval=3)

    assert drifter.effective_dim_context == None
    assert drifter.effective_dim_action_context == None

    actual_context_coef, _, _ = drifter.get_coefficients(
        n_rounds=4,
        effective_dim_context=effective_dim_context,
        effective_dim_action_context=effective_dim_action_context,
    )

    assert drifter.effective_dim_context == 4
    assert drifter.effective_dim_action_context == 3


def test_coefficient_tracker_raises_when_effective_dimensions_are_being_changed():
    effective_dim_context = 4
    effective_dim_action_context = 3
    drifter = CoefficientDrifter(
        drift_interval=3,
        effective_dim_context=effective_dim_context,
        effective_dim_action_context=effective_dim_action_context,
    )

    assert drifter.effective_dim_context == 4
    assert drifter.effective_dim_action_context == 3

    with pytest.raises(
        RuntimeError, match=r"Trying to change the effective dimensions"
    ):
        actual_context_coef, _, _ = drifter.get_coefficients(
            n_rounds=4, effective_dim_context=5, effective_dim_action_context=6
        )


def test_coefficient_tracker_can_shift_coefficient_multiple_times_instantly_based_on_configured_interval():
    effective_dim_context = 4
    effective_dim_action_context = 3

    with mock.patch(
        "obp.dataset.synthetic.sample_random_uniform_coefficients",
        MockCoefSample().fake_sample,
    ):
        drifter = CoefficientDrifter(
            drift_interval=2,
            effective_dim_context=effective_dim_context,
            effective_dim_action_context=effective_dim_action_context,
        )

        actual_context_coef, _, _ = drifter.get_coefficients(n_rounds=5)

    expected_context_coef = np.asarray(
        [
            [2.0, 2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0, 2.0],  # AFTER THIS ROUND, THE COEFS SHOULD CHANGE
            [3.0, 3.0, 3.0, 3.0],
            [3.0, 3.0, 3.0, 3.0],  # AFTER THIS ROUND, THE COEFS SHOULD CHANGE AGAIN
            [4.0, 4.0, 4.0, 4.0],
        ]
    )

    assert np.allclose(actual_context_coef, expected_context_coef)


def test_coefficient_tracker_keeps_track_of_shifted_coefficient_based_on_configured_interval_between_batches():
    effective_dim_context = 4
    effective_dim_action_context = 3

    with mock.patch(
        "obp.dataset.synthetic.sample_random_uniform_coefficients",
        MockCoefSample().fake_sample,
    ):
        drifter = CoefficientDrifter(
            drift_interval=2,
            effective_dim_context=effective_dim_context,
            effective_dim_action_context=effective_dim_action_context,
        )

        actual_context_coef, _, _ = drifter.get_coefficients(n_rounds=3)

        expected_context_coef = np.asarray(
            [
                [2.0, 2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0, 2.0],  # AFTER THIS ROUND, THE COEFS SHOULD CHANGE
                [3.0, 3.0, 3.0, 3.0],
            ]
        )

        assert np.allclose(actual_context_coef, expected_context_coef)

        actual_context_coef, _, _ = drifter.get_coefficients(n_rounds=3)

        expected_context_coef_2 = np.asarray(
            [
                [3.0, 3.0, 3.0, 3.0],  # THIS ROUND SHOULD BE THE SAME AS THE LAST ONE
                [4.0, 4.0, 4.0, 4.0],  # HERE THE COEF SHOULD CHANGE AGAIN
                [4.0, 4.0, 4.0, 4.0],
            ]
        )

    assert np.allclose(actual_context_coef, expected_context_coef_2)


def test_coefficients_can_drift_for_the_action_coefs():
    effective_dim_context = 4
    effective_dim_action_context = 3

    with mock.patch(
        "obp.dataset.synthetic.sample_random_uniform_coefficients",
        MockCoefSample().fake_sample,
    ):
        drifter = CoefficientDrifter(
            drift_interval=2,
            effective_dim_context=effective_dim_context,
            effective_dim_action_context=effective_dim_action_context,
        )

        _, actual_action_coef, _ = drifter.get_coefficients(n_rounds=3)

        expected_action_coef = np.asarray(
            [
                [2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0],  # AFTER THIS ROUND, THE COEFS SHOULD CHANGE
                [3.0, 3.0, 3.0],
            ]
        )

        assert np.allclose(actual_action_coef, expected_action_coef)


def test_coefficients_can_drift_for_the_action_coefs():
    effective_dim_context = 4
    effective_dim_action_context = 3

    with mock.patch(
        "obp.dataset.synthetic.sample_random_uniform_coefficients",
        MockCoefSample().fake_sample,
    ):
        drifter = CoefficientDrifter(
            drift_interval=2,
            effective_dim_context=effective_dim_context,
            effective_dim_action_context=effective_dim_action_context,
        )

        _, _, actual_context_action_coef = drifter.get_coefficients(n_rounds=3)

        expected_context_action_coef = np.asarray([[[2., 2., 2.],
        [2., 2., 2.],
        [2., 2., 2.],
        [2., 2., 2.]],
       [[2., 2., 2.],
        [2., 2., 2.],
        [2., 2., 2.],
        [2., 2., 2.]],
       [[3., 3., 3.],
        [3., 3., 3.],
        [3., 3., 3.],
        [3., 3., 3.]]])

        assert np.allclose(actual_context_action_coef, expected_context_action_coef)

