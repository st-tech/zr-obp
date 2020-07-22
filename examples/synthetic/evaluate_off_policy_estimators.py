import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier as RandomForest
from obp.dataset import (
    SyntheticBanditDataset,
    linear_behavior_policy,
    linear_reward_function
)
from obp.simulator import run_bandit_simulation
from obp.policy import (
    Random,
    BernoulliTS,
    LogisticEpsilonGreedy,
    LogisticTS,
    LogisticUCB
)
from obp.ope import (
    RegressionModel,
    OffPolicyEvaluation,
    InverseProbabilityWeighting,
    SelfNormalizedInverseProbabilityWeighting,
    DirectMethod,
    DoublyRobust,
    SelfNormalizedDoublyRobust,
    SwitchDoublyRobust
)
from obp.utils import estimate_confidence_interval_by_bootstrap


counterfactual_policy_dict = dict(
    bts=BernoulliTS,
    random=Random,
    logistic_egreedy=LogisticEpsilonGreedy,
    logistic_ts=LogisticTS,
    logistic_ucb=LogisticUCB
)

# hyperparameters of the regression model
hyperparams = dict(n_estimators=10, random_state=12345)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate off-policy estimators with synthetic data.')
    parser.add_argument('--n_runs', type=int, default=1,
                        help='number of simulations in the experiment.')
    parser.add_argument('--n_rounds', type=int, default=10000,
                        help='number of rounds for synthetic bandit feedback.')
    parser.add_argument('--n_actions', type=int, default=10,
                        help='number of actions for synthetic bandit feedback.')
    parser.add_argument('--dim_context', type=int, default=5,
                        help='dimensions of context vectors characterizing each round.')
    parser.add_argument('--dim_action_context', type=int, default=5,
                        help='dimensions of context vectors characterizing each action.')
    parser.add_argument('--counterfactual_policy', type=str, required=True,
                        choices=['bts', 'random', 'logistic_ts', 'logistic_ucb', 'logistic_egreedy'],
                        help='counterfacutual policy, bts, random, logistic_ts, logistic_ucb, or logistic_egreedy.')
    parser.add_argument('--random_state', type=int, default=12345)
    args = parser.parse_args()
    print(args)

    n_runs = args.n_runs
    n_rounds = args.n_rounds
    n_actions = args.n_actions
    dim_context = args.dim_context
    dim_action_context = args.dim_action_context
    counterfactual_policy = args.counterfactual_policy
    random_state = args.random_state

    dataset = SyntheticBanditDataset(
        n_actions=n_actions,
        dim_context=dim_context,
        dim_action_context=dim_action_context,
        reward_function=linear_reward_function,
        behavior_policy_function=linear_behavior_policy,
        random_state=random_state
    )

    # hyparparameters for counterfactual policies
    kwargs = dict(
        n_actions=dataset.n_actions,
        len_list=dataset.len_list,
        random_state=random_state
    )
    if 'logistic' in counterfactual_policy:
        kwargs['dim'] = dim_context
    if counterfactual_policy in ['logistic_ucb', 'logistic_egreedy']:
        kwargs['epsilon'] = 0.01
    policy = counterfactual_policy_dict[counterfactual_policy](**kwargs)
    # compared OPE estimators
    ope_estimators = [
        DirectMethod(),
        InverseProbabilityWeighting(),
        SelfNormalizedInverseProbabilityWeighting(),
        DoublyRobust(),
        SelfNormalizedDoublyRobust(),
        SwitchDoublyRobust()
    ]
    # a base ML model for regression model used in Direct Method and Doubly Robust
    base_model = CalibratedClassifierCV(RandomForest(**hyperparams))

    evaluation_of_ope_results = {est.estimator_name: np.zeros(n_runs) for est in ope_estimators}
    for i in np.arange(n_runs):
        # sample a new set of logged bandit feedback
        bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
        # run a counterfactual bandit algorithm on logged bandit feedback data
        selected_actions = run_bandit_simulation(bandit_feedback=bandit_feedback, policy=policy)
        # estimate the ground-truth policy values of the counterfactual policy
        # using the full expected reward contained in the bandit feedback dictionary
        ground_truth_policy_value =\
            bandit_feedback['expected_reward'][np.arange(n_rounds), selected_actions.flatten()].mean()
        # evaluate the estimation performance of OPE estimators
        ope = OffPolicyEvaluation(
            bandit_feedback=bandit_feedback,
            action_context=dataset.action_context,
            regression_model=RegressionModel(base_model=base_model),
            ope_estimators=ope_estimators
        )
        relative_estimation_errors = ope.evaluate_performance_of_estimators(
            selected_actions=selected_actions,
            ground_truth_policy_value=ground_truth_policy_value
        )
        policy.initialize()
        # store relative estimation errors of OPE estimators at each split
        for estimator_name, relative_estimation_error in relative_estimation_errors.items():
            evaluation_of_ope_results[estimator_name][i] = relative_estimation_error

    # estimate confidence intervals of relative estimation by nonparametric bootstrap method
    evaluation_of_ope_results_with_ci = {est.estimator_name: dict() for est in ope_estimators}
    for estimator_name in evaluation_of_ope_results_with_ci.keys():
        evaluation_of_ope_results_with_ci[estimator_name] = estimate_confidence_interval_by_bootstrap(
            samples=evaluation_of_ope_results[estimator_name],
            random_state=random_state
        )
    evaluation_of_ope_results_df = pd.DataFrame(evaluation_of_ope_results_with_ci).T

    print('=' * 60)
    print(f'random_state={random_state}')
    print('-' * 60)
    print(evaluation_of_ope_results_df)
    print('=' * 60)

    # save results of the evaluation of off-policy estimators in './logs' directory.
    log_path = Path('./logs')
    log_path.mkdir(exist_ok=True, parents=True)
    evaluation_of_ope_results_df.to_csv(log_path / 'comparison_of_ope_estimators.csv')
