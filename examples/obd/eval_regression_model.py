import argparse
from typing import Tuple, Dict
from pathlib import Path
import pickle
import yaml

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score

from obp.dataset import OpenBanditDataset
from obp.utils import estimate_confidence_interval_by_bootstrap
from dataset import OBDWithContextSets


def train_eval_lgbm(dataset: OpenBanditDataset,
                    n_splits: int,
                    hyperparams: Dict) -> Tuple[Dict[int, BaseEstimator], Dict[str, np.ndarray]]:
    """Train and Evaluate LightGBM as a regression model."""
    regression_model_dict = dict()
    performance_dict = {metric: np.zeros(n_splits) for metric in ['auc', 'rce']}
    for seed in np.arange(n_splits):
        train, test = dataset.split_data(random_state=seed)
        Xtr, ytr = train['X_reg'], train['reward']
        Xte, yte = test['X_reg'], test['reward']
        # define classifier and make predictions
        clf = HistGradientBoostingClassifier(**hyperparams)
        calibrated_clf = CalibratedClassifierCV(clf, method='isotonic', cv=2)
        calibrated_clf.fit(Xtr, ytr)
        ypred = calibrated_clf.predict_proba(Xte)[:, 1]
        # evaluate prediction accuracies (AUC and RCE)
        performance_dict['auc'][seed] = roc_auc_score(y_true=yte, y_score=ypred)
        rce_mean = log_loss(y_true=yte, y_pred=np.ones_like(yte) * np.mean(ytr))
        rce_clf = log_loss(y_true=yte, y_pred=ypred)
        performance_dict['rce'][seed] = (rce_mean - rce_clf) / rce_clf
        # save model file
        regression_model_dict[seed] = calibrated_clf

    return regression_model_dict, performance_dict


with open('./conf/lightgbm.yaml', 'rb') as f:
    hyperparams = yaml.safe_load(f)['model']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train regression model with LightGBM')
    parser.add_argument('--n_splits', '-n_s', type=int, default=1)
    parser.add_argument('--behavior_policy', '-b_pol', type=str, choices=['bts', 'random'], required=True)
    parser.add_argument('--campaign', '-camp', type=str, choices=['all', 'men', 'women'], required=True)
    args = parser.parse_args()
    print(args)

    obd = OBDWithContextSets(
        behavior_policy=args.behavior_policy,
        campaign=args.campaign,
        data_path=Path('.').resolve().parents[1] / 'obd')

    regression_model_dict, performance_dict = train_eval_lgbm(
        dataset=obd, n_splits=args.n_splits, hyperparams=hyperparams)

    # save trained regression models
    log_path = Path('./logs') / args.behavior_policy / args.campaign
    log_path.mkdir(exist_ok=True, parents=True)
    with open(log_path / 'model.pkl', mode='wb') as f:
        pickle.dump(regression_model_dict, f)

    # verbose
    print('=' * 25)
    for metric, values in performance_dict.items():
        print(f'{metric.upper()}: {np.round(np.mean(values), 6)}')
    print('=' * 25, '\n')

    # save regression model performance
    regression_model_results = dict()
    for metric, values in performance_dict.items():
        regression_model_results[metric] = estimate_confidence_interval_by_bootstrap(values)
    pd.DataFrame(regression_model_results).T.to_csv(log_path / 'regression_model_performance.csv')
