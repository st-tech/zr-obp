# Examples with Open Bandit Dataset (OBD)

## Description

Here, we use the open bandit dataset and pipeline to implement and evaluate OPE.
Specifically, we evaluate the estimation performances of well-known off-policy estimators using the ground-truth policy value of an evaluation policy, which is calculable with our data.

## Configurations

- [`conf/hyperparams.yaml`](https://github.com/st-tech/zr-obp/tree/master/examples/examples_with_obd/conf/hyperparams.yaml)
  The hyperparameters of the some ML model that are used as the regression model in model dependent OPE estimators such as DM and DR or used as the base model of IPW Learner.


## Evaluating Off-Policy Estimators

We evaluate the estimation performances of off-policy estimators, including Direct Method (DM), Inverse Probability Weighting (IPW), and Doubly Robust (DR).

[`./evaluate_off_policy_estimators.py`](./evaluate_off_policy_estimators.py) implements the evaluation of OPE estimators.

```bash
# run evaluation of OPE estimators with (small size) Open Bandit Dataset
python evaluate_off_policy_estimators.py\
    --n_boot_samples $n_boot_samples\
    --base_model $base_model\
    --evaluation_policy $evaluation_policy\
    --behavior_policy $behavior_policy\
    --campaign $campaign\
    --n_sim_to_compute_action_dist $n_sim_to_compute_action_dist\
    --random_state $random_state
```
where `$n_boot_samples` specifies the number of bootstrap samples to estimate mean and standard deviations of the performance of OPE estimators (i.e., *relative estimation error*).
`$base_model` specifies the base ML model for estimating the reward function, and should be one of `logistic_regression`, `random_forest`, or `lightgbm`.
`$evaluation_policy` and `$behavior_policy` specify the evaluation and behavior policies, respectively.
They should be either 'bts' or 'random'.
`$campaign` specifies the campaign and should be one of 'all', 'men', or 'women'.
`$n_sim_to_compute_action_dist` is the number of monte carlo simulation to compute the action distribution of a given evaluation policy.

For example, the following command compares the estimation performances of the three OPE estimators by using Bernoulli TS as evaluation policy and Random as behavior policy in "All" campaign.

```bash
python evaluate_off_policy_estimators.py\
    --n_boot_samples 10\
    --base_model logistic_regression\
    --evaluation_policy bts\
    --behavior_policy random\
    --campaign all

# relative estimation errors of OPE estimators and their standard deviations.
# our evaluation of OPE procedure suggests that DM performs best among the three OPE estimators, because it has low variance property.
# (Note that this result is with the small sample data, and please use the full size data for a more reasonable experiment)
# ==============================
# random_state=12345
# ------------------------------
#          mean       std
# dm   0.193749  0.149464
# ipw  0.308540  0.211327
# dr   0.290177  0.218492
# ==============================
```

Please visit [Examples with Synthetic Data](https://github.com/st-tech/zr-obp/tree/master/examples/examples_with_synthetic) to try the evaluation of OPE estimators with synthetic bandit datasets.

<!--
Moreover, in [benchmark/ope](https://github.com/st-tech/zr-obp/tree/add_benchmark/benchmark/ope), we performed the benchmark experiments on several OPE estimators using the full size Open Bandit Dataset. -->
