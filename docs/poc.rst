=======================
Proof of Concept (PoC)
=======================

As a proof of concept, we use the dataset and pipeline to implement and evaluate OPE estimators.
First, we find that a well-established estimator fails, suggesting that it is critical to choose an appropriate estimator.
We then select a well-performing estimator and use it to improve the platform's fashion item recommendation.

Formal implementations of the following PoCs are available at `examples <https://github.com/st-tech/zr-obp/blob/master/examples/obd/>`_.

Evaluation of Off-Policy Evaluation
----------------------------------------

As an example of the evaluation of OPE, we select the best off-policy estimator among DM, IPW, and DR.
For this purpose, we empirically evaluate their performance as follows (for each campaign separately):


1. For each of the Random and Bernoulli TS policies, randomly split the data collected by that policy into training (70\%) and test (30\%) sets.
2. Estimate the ground-truth value of each policy :math:`\pi` by the empirical mean of clicks in the test set collected by that policy: :math:`V^{\pi} = (T^{\pi}_{test})^{-1} \sum_{t=1}^{T^{\pi}_{test}} Y_t`, where :math:`T^{\pi}_{test}` is the size of the test set of policy :math:`\pi`.
3. Estimate the policy value of each policy by DM, IPW, and DR with the training set collected by the other policy.
4. Repeat the above process :math:`K=15` times by sampling different training sets.
5. Compare the ground-truth and policy value estimated by the bagging prediction:cite:`Breiman1996`.


We measure each estimator's performance with the *Relative-Estimation Error* defined below:

.. math::
    \text{Relative-Estimation Error of }\hat{V}^{\pi} = \left|  \frac{ \left( K^{-1} \sum_{k=1}^K\hat{V}^{\pi}_k \right) - V^{\pi}}{V^{\pi}} \right|.

where :math:`V^{\pi}` is a ground-truth policy value of :math:`\pi` in a test set.
:math:` K^{-1} \sum_{k=1}^K\hat{V}^{\pi}_k ` is a bagging prediction where :math:`\hat{V}^{\pi}_k` is an estimated policy value with the :math:`k`-th bootstrapped samples. :math:`K=15` is the number of folds.


.. code-block:: python

    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier

    from dataset import OBDWithContextSets
    from obp.simulator import OfflineBanditSimulator
    from obp.policy import BernoulliTS

    # configurations
    n_estimators = 15
    behavior_policy = 'random'
    campaign = 'all'

    # load and preprocess Open Bandit Dataset
    obd = OBDWithContextSets(behavior_policy=behavior_policy, campaign=campaign)

    # define a counterfactual policy (that is to be evaluated)
    policy = BernoulliTS(n_actions=dataset.n_actions, len_list=dataset.len_list)

    # we evaluate the following three OPE estimators
    estimators = ['dm', 'ipw', 'dr']

    # split the dataset into the training and test tests
    train, test = obd.split_data(random_state=12345)
    reward_test = test['reward']

    # define a regression model (used in DM and DR)
    lightgbm = HistGradientBoostingClassifier()

    # bagging prediction for the policy value of the counterfactual policy
    ope_results = {est: np.zeros(n_estimators) for est in estimators}
    for seed in np.arange(n_estimators):
        # run a bandit algorithm on logged bandit feedback
        # and conduct off-policy evaluation by using the result of the simulation
        train_boot = obd.sample_bootstrap(train=train) # bootstrapped samples
        sim = OfflineBanditSimulator(train=train_boot, regression_model=regression_model, X_action=obd.X_action)
        sim.simulate(policy=policy) # run offline bandit simulation
        # off-policy evaluation by three standard estimators
        ope_results['dm'][seed] = sim.direct_method()
        ope_results['ipw'][seed] = sim.inverse_probability_weighting()
        ope_results['dr'][seed] = sim.doubly_robust()

        policy.initialize() # initialize counterfactual policy parameters

    # performance of OPE estimators (relative estimation error)
    ground_truth_of_random = np.mean(reward_test)
    for est_name in estimators:
        estimated_policy_value = np.mean(ope_results[est_name])
        relative_estimation_error_of_est = np.abs((estimated_policy_value - ground_truth_of_random) / ground_truth_of_random)
        print(f'{est_name.upper()}: {relative_estimation_error_of_est}')


Evaluation of Bandit Algorithms
----------------------------------------

We the use our dataset and pipeline to evaluate the policy value of a counterfactual policy.
This will enable the fair evaluations of the online bandit algorithms or off-policy learning methods.

As an example, we evaluate the performance of Logistic epsilon greedy by its predicted policy values relative to that of the behavior policy:

.. math::
    \text{relative-CTR of } \pi =  \frac{K^{-1} \sum_{k=1}^{K} \hat{V}^{\pi}_{DR_k} }{ V^{\pi_{\textit{Bernoulli TS}}} },

where the numerator is the bagging prediction of the performance of a counterfactual policy.
The denominator is the ground-truth performance of the behavior policy policy, which is estimated by the empirical mean of factual clicks using the test sets.


.. code-block:: python

    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier

    from obp.dataset import OBDWithContextSets
    from obp.policy import LogisticEpsilonGreedy
    from obp.simulator import OfflineBanditSimulator


    # configurations
    n_estimators = 15
    epsilon = 0.01
    behavior_policy = 'random'
    campaign = 'all'

    # load and preprocess Open Bandit Dataset
    obd = OBDWithContextSets(behavior_policy=behavior_policy, campaign=campaign)

    kwargs = dict(n_actions=obd.n_actions, len_list=obd.len_list, dim=obd.dim_context, epsilon=epsilon)
    policy = LogisticEpsilonGreedy(**kwargs)
    policy_name = policy.policy_name

    # split dataset into training and test sets
    train, test = obd.split_data(random_state=random_state)
    ground_truth_of_behavior_policy = np.mean(test['reward'])

    # define a regression model
    lightgbm = HistGradientBoostingClassifier(**hyperparams)
    regression_model = CalibratedClassifierCV(lightgbm, method='isotonic', cv=2)
    # bagging prediction for the policy value of the counterfactual policy
    ope_results np.zeros(n_estimators)
    for seed in np.arange(n_estimators):
        # run a bandit algorithm on logged bandit feedback
        # and conduct off-policy evaluation by using the result of the simulation
        train_boot = obd.sample_bootstrap(train=train)
        sim = OfflineBanditSimulator(train=train_boot, regression_model=regression_model, X_action=obd.X_action)
        sim.simulate(policy=policy)
        ope_results[seed] = sim.doubly_robust()

        policy.initialize()

    # estimated policy value relative to that of the behavior policy
    relative_estimated_policy_value = np.mean(ope_results[est_name]) / ground_truth_of_behavior_policy
    print(f'{policy_name}: {relative_estimated_policy_value}')
