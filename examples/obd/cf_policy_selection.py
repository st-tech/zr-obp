import argparse
from pathlib import Path
import yaml

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

from dataset import OBDWithContextSets
from obp.policy import LogisticTS, LogisticEpsilonGreedy, LogisticUCB
from obp.simulator import OfflineBanditSimulator
from obp.utils import estimate_confidence_interval_by_bootstrap


cf_policy_dict = dict(
    logistic_egreedy=LogisticEpsilonGreedy,
    logistic_ts=LogisticTS,
    logistic_ucb=LogisticUCB)


with open('./conf/lightgbm.yaml', 'rb') as f:
    hyperparams = yaml.safe_load(f)['model']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run counterfactual policy search')
    parser.add_argument('--n_splits', '-n_s', type=int, default=1)
    parser.add_argument('--n_estimators', '-n_e', type=int, default=2)
    parser.add_argument('--context_set', '-cont', type=str, choices=['1', '2'], required=True)
    parser.add_argument('--counterfactual_policy', '-cf_pol', type=str,
                        choices=['logistic_egreedy', 'logistic_ts', 'logistic_ucb'], required=True)
    parser.add_argument('--epsilon', '-eps', type=float, default=0.1)
    parser.add_argument('--behavior_policy', '-b_pol', type=str, choices=['bts', 'random'], required=True)
    parser.add_argument('--campaign', '-camp', type=str, choices=['all', 'men', 'women'], required=True)
    args = parser.parse_args()
    print(args)

    n_splits = args.n_splits
    n_estimators = args.n_estimators
    context_set = args.context_set
    counterfactual_policy = args.counterfactual_policy
    epsilon = args.epsilon
    behavior_policy = args.behavior_policy
    campaign = args.campaign
    estimators = ['dm', 'ipw', 'dr']

    obd = OBDWithContextSets(
        behavior_policy=behavior_policy,
        campaign=campaign,
        data_path=Path('.').resolve().parents[1] / 'obd',
        context_set=context_set)

    random_state = 12345
    kwargs = dict(n_actions=obd.n_actions, len_list=obd.len_list, dim=obd.dim_context, epsilon=epsilon)
    policy = cf_policy_dict[counterfactual_policy](**kwargs)
    policy_name = policy.policy_name + '_' + context_set

    np.random.seed(random_state)
    # split dataset into training and test sets
    train, test = obd.split_data(random_state=random_state)
    ground_truth = np.mean(test['reward'])

    # define a regression model
    lightgbm = HistGradientBoostingClassifier(**hyperparams)
    regression_model = CalibratedClassifierCV(lightgbm, method='isotonic', cv=2)
    # bagging prediction for the policy value of the counterfactual policy
    ope_results = {est: np.zeros(n_estimators) for est in estimators}
    for seed in np.arange(n_estimators):
        # run a bandit algorithm on logged bandit feedback
        # and conduct off-policy evaluation by using the result of the simulation
        train_boot = obd.sample_bootstrap(train=train)
        sim = OfflineBanditSimulator(
            train=train_boot, regression_model=regression_model, X_action=obd.X_action)
        sim.simulate(policy=policy)
        ope_results['dm'][seed] = sim.direct_method()
        ope_results['ipw'][seed] = sim.inverse_probability_weighting()
        ope_results['dr'][seed] = sim.doubly_robust()

        policy.initialize()

    # estimated policy value at each train-test split
    print('=' * 25)
    print(f'random_state={random_state}')
    print('-----')
    for est_name in estimators:
        relative_estimated_policy_value = np.mean(ope_results[est_name]) / ground_truth
        print(f'{est_name.upper()}: {np.round(relative_estimated_policy_value, 6)}')
    print('=' * 25, '\n')

    # estimate confidence intervals by nonparametric bootstrap method
    relative_ope_results_with_ci = {est: dict() for est in estimators}
    for est_name in estimators:
        relative_ope_results_with_ci[est_name] = estimate_confidence_interval_by_bootstrap(
            samples=ope_results[est_name] / ground_truth, random_state=random_state)

    # save cf policy selection results
    save_path = Path('./logs') / behavior_policy / campaign / 'cf_policy_selection'
    save_path.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(relative_ope_results_with_ci).T.to_csv(save_path / f'{policy_name}.csv')
