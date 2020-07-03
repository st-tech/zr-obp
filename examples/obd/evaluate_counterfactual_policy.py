import argparse
from pathlib import Path
import yaml

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

from dataset import OBDWithInteractionFeatures
from obp.policy import LogisticTS, LogisticEpsilonGreedy, LogisticUCB
from obp.simulator import run_bandit_simulation
from obp.ope import (
    RegressionModel,
    OffPolicyEvaluation,
    InverseProbabilityWeighting,
    DirectMethod,
    DoublyRobust
)


with open('./conf/lightgbm.yaml', 'rb') as f:
    hyperparams = yaml.safe_load(f)['model']

counterfactual_policy_dict = dict(
    logistic_egreedy=LogisticEpsilonGreedy,
    logistic_ts=LogisticTS,
    logistic_ucb=LogisticUCB
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run off-policy evaluation of a counterfactual bandit policy.')
    parser.add_argument('--context_set', '-cont', type=str, choices=['1', '2'], required=True)
    parser.add_argument('--counterfactual_policy', '-cf_pol', type=str,
                        choices=['logistic_egreedy', 'logistic_ts', 'logistic_ucb'], required=True)
    parser.add_argument('--epsilon', '-eps', type=float, default=0.1)
    parser.add_argument('--behavior_policy', '-b_pol', type=str, choices=['bts', 'random'], required=True)
    parser.add_argument('--campaign', '-camp', type=str, choices=['all', 'men', 'women'], required=True)
    parser.add_argument('--random_state', type=int, default=12345)
    args = parser.parse_args()
    print(args)

    context_set = args.context_set
    counterfactual_policy = args.counterfactual_policy
    epsilon = args.epsilon
    behavior_policy = args.behavior_policy
    campaign = args.campaign
    random_state = args.random_state

    obd = OBDWithInteractionFeatures(
        behavior_policy=behavior_policy,
        campaign=campaign,
        data_path=Path('.').resolve().parents[1] / 'obd',
        context_set=context_set
    )

    kwargs = dict(n_actions=obd.n_actions, len_list=obd.len_list, dim=obd.dim_context, epsilon=epsilon)
    policy = counterfactual_policy_dict[counterfactual_policy](**kwargs)
    policy_name = policy.policy_name + '_' + context_set

    np.random.seed(random_state)
    # split dataset into training and test sets
    train, test = obd.split_data(random_state=random_state)
    # ground-truth policy value of the random policy
    # , which is the empirical mean of the factual (observed) rewards
    ground_truth = np.mean(test['reward'])

    # a base ML model for regression model used in Direct Method and Doubly Robust
    base_model = CalibratedClassifierCV(HistGradientBoostingClassifier(**hyperparams))
    # run a counterfactual bandit algorithm on logged bandit feedback data
    selected_actions = run_bandit_simulation(train=train, policy=policy)
    # estimate the policy value of a given counterfactual algorithm by the three OPE estimators.
    ope = OffPolicyEvaluation(
        train=train,
        regression_model=RegressionModel(base_model=base_model),
        action_context=obd.action_context,
        ope_estimators=[InverseProbabilityWeighting(), DirectMethod(), DoublyRobust()]
    )
    estimated_policy_value, estimated_interval = ope.summarize_off_policy_estimates(selected_actions=selected_actions)

    # estimated policy value at each train-test split
    print('=' * 70)
    print(f'random_state={random_state}: counterfactual policy={policy_name}')
    print('-' * 70)
    estimated_policy_value['relative_estimated_policy_value'] =\
        estimated_policy_value.estimated_policy_value / ground_truth
    print(estimated_policy_value)
    print('=' * 70)

    # save counterfactual policy evaluation results
    save_path = Path('./logs') / behavior_policy / campaign / 'cf_policy_selection'
    save_path.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(estimated_policy_value).to_csv(save_path / f'{policy_name}.csv')
    # save visualization of the off-policy evaluation results
    ope.visualize_off_policy_estimates(
        selected_actions=selected_actions,
        fig_dir=save_path,
        fig_name=f'{policy_name}.png'
    )
