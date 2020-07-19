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
    OffPolicyEvaluation,
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
    parser = argparse.ArgumentParser(description='evaluate off-policy estimators.')
    parser.add_argument('--n_boot_samples', type=int, default=1,
                        help='number of bootstrap samples in the experiment.')
    parser.add_argument('--counterfactual_policy', type=str, choices=['bts', 'random'], required=True,
                        help='counterfacutual policy, bts or random.')
    parser.add_argument('--behavior_policy', type=str, choices=['bts', 'random'], required=True,
                        help='behavior policy, bts or random.')
    parser.add_argument('--campaign', type=str, choices=['all', 'men', 'women'], required=True,
                        help='campaign name, men, women, or all.')
    parser.add_argument('--random_state', type=int, default=12345)
    args = parser.parse_args()
    print(args)

    n_boot_samples = args.n_boot_samples
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

    # hyparparameters for counterfactual policies
    kwargs = dict(n_actions=obd.n_actions, len_list=obd.len_list, random_state=random_state)
    if counterfactual_policy == 'bts':
        kwargs['alpha'] = production_prior_for_bts[campaign]['alpha']
        kwargs['beta'] = production_prior_for_bts[campaign]['beta']
        kwargs['batch_size'] = production_batch_size_for_bts[campaign]
    policy = counterfactual_policy_dict[counterfactual_policy](**kwargs)
    # compared OPE estimators
    ope_estimators = [
        DirectMethod(),
        InverseProbabilityWeighting(),
        DoublyRobust()
    ]
    # a base ML model for regression model used in Direct Method and Doubly Robust
    base_model = CalibratedClassifierCV(HistGradientBoostingClassifier(**hyperparams))
    # ground-truth policy value of a counterfactual policy
    # , which is estimated with factual (observed) rewards (on-policy estimation)
    ground_truth_policy_value = OpenBanditDataset.calc_on_policy_policy_value_estimate(
        behavior_policy=counterfactual_policy,
        campaign=campaign,
        data_path=data_path
    )

    evaluation_of_ope_results = {est.estimator_name: np.zeros(n_boot_samples) for est in ope_estimators}
    for b in np.arange(n_boot_samples):
        # sample bootstrap from batch logged bandit feedback
        boot_bandit_feebdack = obd.sample_bootstrap_bandit_feedback(random_state=b)
        # run a counterfactual bandit algorithm on logged bandit feedback data
        selected_actions = run_bandit_simulation(bandit_feedback=boot_bandit_feebdack, policy=policy)
        # evaluate the estimation performance of OPE estimators
        ope = OffPolicyEvaluation(
            bandit_feedback=boot_bandit_feebdack,
            action_context=obd.action_context,
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
            evaluation_of_ope_results[estimator_name][b] = relative_estimation_error

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

    # save results of the evaluation of off-policy estimators in './logs' directory.
    log_path = Path('./logs') / behavior_policy / campaign
    log_path.mkdir(exist_ok=True, parents=True)
    evaluation_of_ope_results_df.to_csv(log_path / 'comparison_of_ope_estimators.csv')
