=======================
Proof of Concept (PoC)
=======================

As a proof of concept, we use the dataset and pipeline to implement and evaluate OPE estimators.
We first evaluate well-known off-policy estimators with the ground-truth performance of a counterfactual policy.
We then use these OPE estimators to evaluate a counterfactual logistic bandit policies.

Formal implementations of the following PoCs are available at `examples <https://github.com/st-tech/zr-obp/blob/master/examples/obd/>`_.

Evaluation of Off-Policy Evaluation
----------------------------------------

As an example of the evaluation of OPE, we select the best off-policy estimator among DM, IPW, and DR.
For this purpose, we empirically evaluate their performance as follows (for each campaign separately):

1. For each of the Random and Bernoulli TS policies, independently sample the data collected by that policy *with replacement* (bootstrap sampling).
2. Estimate the ground-truth value of each policy :math:`\pi` by the empirical mean of clicks collected by that policy: :math:`V^{\pi} = (T^{\pi})^{-1} \sum_{t=1}^{T^{\pi}} Y_t`, where :math:`T^{\pi}` is the size of bandit feedback of policy :math:`\pi`.
3. Estimate the policy value of each policy by DM, IPW, and DR with the bootstrapped logged bandit feedback collected by the other policy.
4. Repeat the above process :math:`K` times with different bootstrapped samples.
5. Compare the ground-truth and estimated policy values by OPE estimators.

We measure each estimator's performance with the *Relative-Estimation Error* defined below:

.. math::
    \text{Relative-Estimation Error of }\hat{V}^{\pi} = \left|  \frac{ \hat{V}^{\pi}_k - V^{\pi}}{V^{\pi}} \right|.

where :math:`V^{\pi}` is a ground-truth policy value of :math:`\pi`, which is estimated by the empirical mean of factual clicks in the logged bandit feedback (on-policy estimate).
:math:`\hat{V}^{\pi}_k` is an estimated policy value with the :math:`k`-th bootstrapped samples.
We use estimated policy values :math:`\{ \hat{V}^{\pi}_k \}_{k=1}^K` to nonparametrically estimate confidence intervals of the performance of OPE estimators.


For example, the following code compares the estimation performance of the three OPE estimators
by using Bernoulli TS as counterfactual policy and Random as behavior policy in "All" campaign.

.. code-block:: python

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

    # hyperparameter for the regression model (LightGBM)
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

    # configurations for the evaluation of OPE estimators
    n_boot_samples = 10
    counterfactual_policy = 'bts'
    behavior_policy = 'random'
    campaign = 'all'
    random_state = 12345
    data_path = Path('.').resolve().parents[1] / 'obd'

    # load and preprocess Open Bandit Dataset
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
        # run a counterfactual bandit algorithm on the bootstrapped logged bandit feedback data
        selected_actions = run_bandit_simulation(bandit_feedback=boot_bandit_feebdack, policy=policy)
        # evaluate the estimation performance of OPE estimators by relative estimation error
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

        # store relative estimation errors of OPE estimators at each bootstrap sample
        for estimator_name, relative_estimation_error in relative_estimation_errors.items():
            evaluation_of_ope_results[estimator_name][b] = relative_estimation_error

    # estimate confidence intervals of relative estimation by nonparametric bootstrap method
    evaluation_of_ope_results_with_ci = {est.estimator_name: dict() for est in ope_estimators}
    for estimator_name in evaluation_of_ope_results_with_ci.keys():
        evaluation_of_ope_results_with_ci[estimator_name] = estimate_confidence_interval_by_bootstrap(
            samples=evaluation_of_ope_results[estimator_name],
            random_state=random_state
        )

    print('=' * 50)
    print(f'random_state={random_state}')
    print('-' * 50)
    print(pd.DataFrame(evaluation_of_ope_results_with_ci).T)
    print('=' * 50)

    # relative estiamtion errors and their 95% confidence intervals of OPE estimators.
    # our evaluation of OPE procedure suggests that DM performs best among the three OPE estimators because DM has low variance property.
    # (Note that this result is with the small sample data and please see our paper for the results with the full size data)
    # ==================================================
    # random_state=12345
    # --------------------------------------------------
    #          mean  95.0% CI (lower)  95.0% CI (upper)
    # dm   0.218148           0.14561           0.29018
    # ipw  1.158730           0.96190           1.53333
    # dr   0.992942           0.71789           1.35594
    # ==================================================


If you want to run the above experiment to evaluate OPE estimators, please see `examples <https://github.com/st-tech/zr-obp/blob/master/examples/obd/>`_.


Evaluation of Bandit Algorithms
----------------------------------------

We then use our dataset and pipeline to evaluate the policy value of a counterfactual logistic bandit policy.

As an example, we evaluate the performance of Logistic Upper Confidence Bound (logistic ucb) by its predicted policy values by OPE estimators relative to that of the behavior policy:

.. math::
    \text{relative-CTR of } \pi =  \hat{V}^{\pi} / V^{\pi_{\textit{behavior}}} ,

where the numerator is the estimated performance of a counterfactual policy.
The denominator is the ground-truth performance of the behavior policy, which is estimated by the empirical mean of factual clicks in the logged bandit feedback (on-policy estimate).

For example, the following code evaluates the performance of the logistic_ucb policy (context_set='1' and exploration hyperparameter=0.1)
by using the three OPE estimators and Random as behavior policy in "All" campaign.

.. code-block:: python

    from pathlib import Path
    import yaml

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

    # hyperparameter for the regression model (LightGBM)
    with open('./conf/lightgbm.yaml', 'rb') as f:
    hyperparams = yaml.safe_load(f)['model']

    counterfactual_policy_dict = dict(
        logistic_egreedy=LogisticEpsilonGreedy,
        logistic_ts=LogisticTS,
        logistic_ucb=LogisticUCB
    )

    # configurations
    context_set = '1'
    counterfactual_policy = 'logistic_ucb'
    epsilon = 0.1
    behavior_policy = 'random'
    campaign = 'all'
    random_state = 12345
    data_path = Path('.').resolve().parents[1] / 'obd'

    obd = OBDWithInteractionFeatures(
        behavior_policy=behavior_policy,
        campaign=campaign,
        data_path=data_path,
        context_set=context_set
    )

    # hyperparameters for logistic bandit policies
    kwargs = dict(
        n_actions=obd.n_actions,
        len_list=obd.len_list,
        dim=obd.dim_context,
        random_state=random_state
    )
    if counterfactual_policy != 'logistic_ts':
        kwargs['epsilon'] = epsilon
    policy = counterfactual_policy_dict[counterfactual_policy](**kwargs)
    policy_name = f'{policy.policy_name}_{context_set}'

    # obtain batch logged bandit feedback generated by behavior policy
    bandit_feedback = obd.obtain_batch_bandit_feedback()
    # ground-truth policy value of the random policy
    # , which is the empirical mean of the factual (observed) rewards
    ground_truth = bandit_feedback['reward'].mean()

    # a base ML model for regression model used in Direct Method and Doubly Robust
    base_model = CalibratedClassifierCV(HistGradientBoostingClassifier(**hyperparams))

    # run a counterfactual bandit algorithm on logged bandit feedback data
    selected_actions = run_bandit_simulation(bandit_feedback=bandit_feedback, policy=policy)
    # estimate the policy value of a given counterfactual algorithm by the three OPE estimators.
    ope = OffPolicyEvaluation(
        bandit_feedback=bandit_feedback,
        regression_model=RegressionModel(base_model=base_model),
        action_context=obd.action_context,
        ope_estimators=[InverseProbabilityWeighting(), DirectMethod(), DoublyRobust()]
    )
    estimated_policy_value, estimated_interval = ope.summarize_off_policy_estimates(selected_actions=selected_actions)

    # estimated policy value and that realtive to that of the behavior policy
    print('=' * 70)
    print(f'random_state={random_state}: counterfactual policy={policy_name}')
    print('-' * 70)
    estimated_policy_value['relative_estimated_policy_value'] = estimated_policy_value.estimated_policy_value / ground_truth
    print(estimated_policy_value)
    print('=' * 70)

    # estimated policy values relative to the behavior policy (the Random policy) of a counterfactual policy (logistic UCB with Context Set 1)
    # by three OPE estimators (IPW: inverse probability weighting, DM; Direct Method, DR; Doubly Robust)
    # in this example, DM predicts that the counterfactual policy outperforms the behavior policy by about 2.59%
    # (Note that this result is with the small sample data and please see our paper for the results with the full size data)
    # ======================================================================
    # random_state=12345: counterfactual policy=logistic_ucb_0.1_1
    # ----------------------------------------------------------------------
    #      estimated_policy_value  relative_estimated_policy_value
    # ipw                0.008000                         2.105263
    # dm                 0.003898                         1.025915
    # dr                 0.007948                         2.091689
    # ======================================================================


If you want to run the above experiment to evaluate counterfactual logistic bandit policies, please see `examples <https://github.com/st-tech/zr-obp/blob/master/examples/obd/>`_.
