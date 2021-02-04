# Example with the Open Bandit Dataset (OBD)

## Description

Here, we use the open bandit dataset and pipeline to implement and evaluate OPE. Specifically, we evaluate the estimation performances of well-known off-policy estimators using the ground-truth policy value of an evaluation policy, which is calculable with our data using on-policy estimation.

## Evaluating Off-Policy Estimators

We evaluate the estimation performances of off-policy estimators, including Direct Method (DM), Inverse Probability Weighting (IPW), and Doubly Robust (DR).

### Files
- [`./evaluate_off_policy_estimators.py`](./evaluate_off_policy_estimators.py) implements the evaluation of OPE estimators.
- [`.conf/hyperparams.yaml`](./conf/hyperparams.yaml) defines hyperparameters of some machine learning models used as the regression model in model dependent estimators (such as DM and DR).

### Scripts

```bash
# run evaluation of OPE estimators with (small size) Open Bandit Dataset
python evaluate_off_policy_estimators.py\
    --n_runs $n_runs\
    --base_model $base_model\
    --evaluation_policy $evaluation_policy\
    --behavior_policy $behavior_policy\
    --campaign $campaign\
    --n_sim_to_compute_action_dist $n_sim_to_compute_action_dist\
    --n_jobs $n_jobs\
    --random_state $random_state
```
- `$n_runs` specifies the number of bootstrap sampling to estimate means and standard deviations of the performance of OPE estimators (i.e., relative estimation error).
- `$base_model` specifies the base ML model for estimating the reward function, and should be one of `logistic_regression`, `random_forest`, or `lightgbm`.
- `$evaluation_policy` and `$behavior_policy` specify the evaluation and behavior policies, respectively.
They should be either 'bts' or 'random'.
- `$campaign` specifies the campaign and should be one of 'all', 'men', or 'women'.
- `$n_sim_to_compute_action_dist` is the number of monte carlo simulation to compute the action distribution of a given evaluation policy.
- `$n_jobs` is the maximum number of concurrently running jobs.

For example, the following command compares the estimation performances of the three OPE estimators by using Bernoulli TS as evaluation policy and Random as behavior policy in "All" campaign.

```bash
python evaluate_off_policy_estimators.py\
    --n_runs 20\
    --base_model logistic_regression\
    --evaluation_policy bts\
    --behavior_policy random\
    --campaign all\
    --n_jobs -1

# relative estimation errors of OPE estimators and their standard deviations.
# our evaluation of OPE procedure suggests that DM performs best among the three OPE estimators, because it has low variance property.
# (Note that this result is with the small sample data, and please use the full size data for a more reasonable experiment)
# ==============================
# random_state=12345
# ------------------------------
#          mean       std
# dm   0.180288  0.114694
# ipw  0.333113  0.350425
# dr   0.304401  0.347842
# ==============================
```

Please refer to [this page](https://zr-obp.readthedocs.io/en/latest/evaluation_ope.html) for the evaluation of OPE protocol using our real-world data.
Please visit [synthetic](../synthetic/) to try the evaluation of OPE estimators with synthetic bandit datasets.
Moreover, in [benchmark/ope](https://github.com/st-tech/zr-obp/tree/master/benchmark/ope), we performed the benchmark experiments on several OPE estimators using the full size Open Bandit Dataset.
