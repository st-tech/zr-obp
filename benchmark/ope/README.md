# Benchmarking Off-Policy Evaluation

## Description
We use the (full size) open bandit dataset to evaluate and compare OPE estimators in a *realistic* and *reproducible* manner. Specifically, we evaluate the estimation performances of a wide variety of existing estimators by comparing the estimated policy values with the ground-truth of an evaluation policy contained in the data.

### Dataset
Please download the full [open bandit dataset](https://research.zozo.com/data.html) and put it as the `../open_bandit_dataset/` directory.

## Training Regression Model

Model-dependent estimators such as DM and DR need a pre-trained regression model.
Here, we train a regression model with some machine learning methods.

[train_regression_model.py](https://github.com/st-tech/zr-obp/blob/master/benchmark/ope/train_regression_model.py) implements the training process of the regression model. ([`conf/hyperparams.yaml`](https://github.com/st-tech/zr-obp/blob/master/benchmark/ope/conf/hyperparams.yaml) defines hyperparameters for the machine learning methods.)

```
python train_regression_model.py\
    --n_runs $n_runs\
    --base_model $base_model\ "logistic_regression" or "lightgbm"
    --behavior_policy $behavior_policy\ "random" or "bts"
    --campaign $campaign\ # "men", "women", or "all"
    --n_sim_to_compute_action_dist $n_sim_to_compute_action_dist\
    --is_timeseries_split $is_timeseries_split\ # in-sample or out-sample
    --test_size $test_size\
    --is_mrdr $is_mrdr\ # use "more robust doubly robust" option or not
    --n_jobs $n_jobs\
    --random_state $random_state
```

where
- `$n_runs` specifies the number of simulation runs with different bootstrap samples in the experiment.
- `$base_model` specifies the base ML model for defining the regression model and should be one of "logistic_regression", "random_forest", or "lightgbm".
- `$campaign` specifies the campaign considered in ZOZOTOWN and should be one of "all", "men", or "women".
- `$n_sim_to_compute_action_dist` is the number of monte carlo simulation to compute the action choice probabilities by a given evaluation policy.
- `$is_timeseries_split` is whether the data is split based on timestamp or not. If true, the out-sample performance of OPE is tested. See the relevant paper for details.
- `$test_size` specifies the proportion of the dataset to include in the test split when `$is_timeseries_split=True`.
- `$is_mrdr` is whether the regression model is trained by the more robust doubly robust way. See the relevant paper for details.
- `$n_jobs` is the maximum number of concurrently running jobs.

For example, the following command trains the regression model based on logistic regression on the logged bandit feedback data collected by the Random policy (as a behavior policy) in "All" campaign.

```bash
python train_regression_model.py\
    --n_runs 10\
    --base_model logistic_regression\
    --behavior_policy random\
    --campaign all\
    --is_mrdr False\
    --is_timeseries_split False
```


## Evaluating Off-Policy Estimators

Next, we evaluate and compare the estimation performances of the following OPE estimators:

- Direct Method (DM)
- Inverse Probability Weighting (IPW)
- Self-Normalized Inverse Probability Weighting (SNIPW)
- Doubly Robust (DR)
- Self-Normalized Doubly Robust (SNDR)
- Switch Doubly Robust (Switch-DR)
- Doubly Robust with Optimistic Shrinkage (DRos)
-  More Robust Doubly Robust (MRDR)

For Switch-DR and DRos, we test some different values of hyperparameters.
See our [documentation](https://zr-obp.readthedocs.io/en/latest/estimators.html) for the details about these estimators.


[benchmark_off_policy_estimators.py](https://github.com/st-tech/zr-obp/blob/master/benchmark/ope/benchmark_off_policy_estimators.py) implements the evaluation and comparison of OPE estimators using the open bandit dataset.
Note that you have to finish training a regression model (see the above section) before conducting the evaluation of OPE in the corresponding setting.
We summarize the detailed experimental protocol for evaluating OPE estimators using real-world data [here](https://zr-obp.readthedocs.io/en/latest/evaluation_ope.html).

```
# run evaluation of OPE estimators with the full open bandit dataset
python benchmark_off_policy_estimators.py\
    --n_runs $n_runs\
    --base_model $base_model\ "logistic_regression" or "lightgbm"
    --behavior_policy $behavior_policy\ "random" or "bts"
    --campaign $campaign\ # "men", "women", or "all"
    --n_sim_to_compute_action_dist $n_sim_to_compute_action_dist\
    --is_timeseries_split\ # in-sample or out-sample
    --test_size $test_size\
    --n_jobs $n_jobs\
    --random_state $random_state
```
where
- `$n_runs` specifies the number of simulation runs with different bootstrap samples in the experiment to estimate standard deviations of the performance of OPE estimators.
- $base_model_for_evaluation_policy specifies the base ML model for defining the regression model and should be one of "logistic_regression", "random_forest", or "lightgbm".
- `$campaign` specifies the campaign considered in ZOZOTOWN and should be one of "all", "men", or "women".
- `$n_sim_to_compute_action_dist` is the number of monte carlo simulation to compute the action choice probabilities by a given evaluation policy.
- `$is_timeseries_split` is whether the data is split based on timestamp or not. If true, the out-sample performance of OPE is tested. See the relevant paper for details.
- `$test_size` specifies the proportion of the dataset to include in the test split when `$is_timeseries_split=True`.
- `$n_jobs` is the maximum number of concurrently running jobs.

For example, the following command compares the estimation performances of the OPE estimators listed above using Bernoulli TS as an evaluation policy and Random as a behavior policy in "All" campaign in the out-sample situation.

```bash
python benchmark_off_policy_estimators.py\
    --n_runs 10\
    --base_model logistic_regression\
    --behavior_policy random\
    --campaign all\
    --test_size 0.3\
    --is_timeseries_split True
```

The results of our benchmark experiments can be found in Section 5 of [our paper](https://arxiv.org/abs/2008.07146).
