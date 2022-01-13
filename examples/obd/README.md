# Example Experiment with Open Bandit Dataset

## Description

We use Open Bandit Dataset to implement the evaluation of OPE. Specifically, we evaluate the estimation performance of some well-known OPE estimators using the on-policy policy value of an evaluation policy, which is calculable with the dataset.

## Evaluating Off-Policy Estimators

In the following, we evaluate the estimation performance of

- Direct Method (DM)
- Inverse Probability Weighting (IPW)
- Self-Normalized Inverse Probability Weighting (SNIPW)
- Doubly Robust (DR)
- Self-Normalized Doubly Robust (SNDR)
- Switch Doubly Robust (Switch-DR)
- Doubly Robust with Optimistic Shrinkage (DRos)

For Switch-DR and DRos, we tune the built-in hyperparameters using SLOPE, a data-driven hyperparameter tuning method for OPE estimators.
See [our documentation](https://zr-obp.readthedocs.io/en/latest/estimators.html) for the details about these estimators.

### Files
- [`./evaluate_off_policy_estimators.py`](./evaluate_off_policy_estimators.py) implements the evaluation of OPE estimators using Open Bandit Dataset.
- [`.conf/hyperparams.yaml`](./conf/hyperparams.yaml) defines hyperparameters of some ML models used as the regression model in model dependent estimators (such as DM and DR).

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

For example, the following command compares the estimation performance of the three OPE estimators by using Bernoulli TS as evaluation policy and Random as behavior policy in "All" campaign.

```bash
python evaluate_off_policy_estimators.py\
    --n_runs 30\
    --base_model logistic_regression\
    --evaluation_policy bts\
    --behavior_policy random\
    --campaign all\
    --n_jobs -1

# relative estimation errors of OPE estimators and their standard deviations.
# ==============================
# random_state=12345
# ------------------------------
#                mean       std
# dm         0.156876  0.109898
# ipw        0.311082  0.311170
# snipw      0.311795  0.334736
# dr         0.292464  0.315485
# sndr       0.302407  0.328434
# switch-dr  0.258410  0.160598
# dr-os      0.159520  0.109660
# ==============================
```

Please refer to [this page](https://zr-obp.readthedocs.io/en/latest/evaluation_ope.html) for the evaluation of OPE protocol using our real-world data. Please visit [synthetic](../synthetic/) to try the evaluation of OPE estimators with synthetic bandit data. Moreover, in [benchmark/ope](https://github.com/st-tech/zr-obp/tree/master/benchmark/ope), we performed the benchmark experiments on several OPE estimators using the full size Open Bandit Dataset.
