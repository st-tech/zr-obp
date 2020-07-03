import argparse
from pathlib import Path
import yaml

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

from obp.dataset import OpenBanditDataset
from obp.simulator import run_bandit_simulation
from obp.policy import Random, BernoulliTS
from obp.ope import (
    RegressionModel,
    CompareOffPolicyEstimators,
    InverseProbabilityWeighting,
    DirectMethod,
    DoublyRobust
)
from obp.utils import estimate_confidence_interval_by_bootstrap


with open('./conf/lightgbm.yaml', 'rb') as f:
    hyperparams = yaml.safe_load(f)['model']

# configurations to reproduce the Bernoulli Thompson Sampling policy
# used in ZOZOTOWN production
with open('./conf/prior_bts.yaml', 'rb') as f:
    production_prior_for_bts = yaml.safe_load(f)

with open('./conf/batch_size_bts.yaml', 'rb') as f:
    production_batch_size_for_bts = yaml.safe_load(f)

counterfactual_policy_dict = dict(
    bts=BernoulliTS,
    random=Random
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate off-policy estimators')
    parser.add_argument('--n_splits', '-n_s', type=int, default=1)
    parser.add_argument('--counterfactual_policy', '-c_pol', type=str, choices=['bts', 'random'], required=True)
    parser.add_argument('--behavior_policy', '-b_pol', type=str, choices=['bts', 'random'], required=True)
    parser.add_argument('--campaign', '-camp', type=str, choices=['all', 'men', 'women'], required=True)
    parser.add_argument('--random_state', type=int, default=12345)
    args = parser.parse_args()
    print(args)

    n_splits = args.n_splits
    counterfactual_policy = args.counterfactual_policy
    behavior_policy = args.behavior_policy
    campaign = args.campaign
    random_state = args.random_state
    data_path = Path('.').resolve().parents[1] / 'obd'

    obd = OpenBanditDataset(
        behavior_policy=behavior_policy,
        campaign=campaign,
        data_path=data_path
    )

    kwargs = dict(n_actions=obd.n_actions, len_list=obd.len_list, random_state=random_state)
    if behavior_policy == 'random':
        kwargs['alpha'] = production_prior_for_bts[campaign]['alpha']
        kwargs['beta'] = production_prior_for_bts[campaign]['beta']
        kwargs['batch_size'] = production_batch_size_for_bts[campaign]
    policy = counterfactual_policy_dict[behavior_policy](**kwargs)
    # compared OPE estimators
    ope_estimators = [
        DirectMethod(),
        InverseProbabilityWeighting(),
        DoublyRobust()
    ]
    # a base ML model for regression model used in Direct Method and Doubly Robust
    base_model = CalibratedClassifierCV(HistGradientBoostingClassifier(**hyperparams))

    evaluation_of_ope_results = {est.estimator_name: np.zeros(n_splits) for est in ope_estimators}
    for s in np.arange(n_splits):
        # split dataset into training and test sets
        train, test = obd.split_data(test_size=0.3, random_state=s)
        # run a counterfactual bandit algorithm on logged bandit feedback data
        selected_actions = run_bandit_simulation(train=train, policy=policy)
        # evaluate the estimation performance of OPE estimators
        compare_ope = CompareOffPolicyEstimators(
            train=train,
            factual_rewards=test['reward'],
            action_context=obd.action_context,
            regression_model=RegressionModel(base_model=base_model),
            ope_estimators=ope_estimators
        )
        relative_estimation_errors = compare_ope.evaluate_performance_of_estimators(
            selected_actions=selected_actions
        )
        policy.initialize()
        # store relative estimation errors of OPE estimators at each split
        for estimator_name, relative_estimation_error in relative_estimation_errors.items():
            evaluation_of_ope_results[estimator_name][s] = relative_estimation_error

    # estimate confidence intervals of relative estimation by nonparametric bootstrap method
    evaluation_of_ope_results_with_ci = {est.estimator_name: dict() for est in ope_estimators}
    for estimator_name in evaluation_of_ope_results_with_ci.keys():
        evaluation_of_ope_results_with_ci[estimator_name] = estimate_confidence_interval_by_bootstrap(
            samples=evaluation_of_ope_results[estimator_name],
            random_state=random_state
        )
    evaluation_of_ope_results_df = pd.DataFrame(evaluation_of_ope_results_with_ci).T

    print('=' * 50)
    print(f'random_state={random_state}')
    print('-' * 50)
    print(evaluation_of_ope_results_df)
    print('=' * 50)

    # save results of evaluation of off-policy estimators
    log_path = Path('./logs') / behavior_policy / campaign
    evaluation_of_ope_results_df.to_csv(log_path / 'comparison_of_ope_estimators.csv')
