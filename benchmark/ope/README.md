# Benchmarking Off-Policy Evaluation

## Description
We use the (full size) open bandit dataset to evaluate and compare OPE estimators in a *realistic* and *reproducible* manner. Specifically, we evaluate the estimation performances of a wide variety of existing estimators by comparing the estimated policy values with the ground-truth of an evaluation policy contained in the data.

### Dataset
Please download the full [open bandit dataset](https://research.zozo.com/data.html) and put it as the `../open_bandit_dataset/` directory.

## Training Regression Model

Here we train a regression model using some machine learning methods.
We define hyperparameters for the machine learning methods in [`conf/hyperparams.yaml`](https://github.com/st-tech/zr-obp/blob/master/benchmark/ope/conf/hyperparams.yaml).
This will be used in the model-dependent estimators such.
[train_regression_model.py](https://github.com/st-tech/zr-obp/blob/master/benchmark/ope/train_regression_model.py) implements the training process of the regression model.

```
python train_regression_model.py\
    --n_runs $n_runs\
    --base_model $base_model\ "logistic_regression" or "lightgbm"
    --behavior_policy $behavior_policy\ "random" or "bts"
    --campaign $campaign\ # "men", "women", or "all"
    --is_timeseries_split $is_timeseries_split\ # in-sample or out-sample
    --n_sim_to_compute_action_dist $n_sim_to_compute_action_dist\
    --is_mrdr $is_mrdr\ # use "more robust doubly robust" option or not
    --n_jobs $n_jobs\
    --random_state $random_state
```

where
- `$n_runs` specifies the number of simulation runs with different bootstrap samples in the experiment.
- `$base_model` specifies the base ML model for defining the regression model and should be one of "logistic_regression" or "lightgbm".
- `$campaign` specifies the campaign and should be one of 'all', 'men', or 'women'.
- `$n_sim_to_compute_action_dist` is the number of monte carlo simulation to compute the action distribution of a given evaluation policy.
- `$test_size` specifies the proportion of the dataset to include in the test split (when `$is_timeseries_split` is applied.)
- `$is_timeseries_split` is whether the data is split based on timestamp or not. If true, the out-sample performance of OPE is tested. See the relevant paper for details.
- `$is_mrdr` is whether the regression model is trained by the more robust doubly robust way or not. See the relevant paper for details.
- `$n_jobs` is the maximum number of concurrently running jobs.

For example, the following command trains the regression model based on logistic regression on the logged bandit feedback data collected by the Random policy (as behavior policy) in "All" campaign by the more robust doubly robust way.

```bash
python train_regression_model.py\
    --n_runs 10\
    --base_model logistic_regression\
    --behavior_policy random\
    --campaign all\
    --is_mrdr True\
    --is_timeseries_split False
```


```
for model in random_forest
do
    for pi_b in random
    do
        for camp in men
        do
            for is_mrdr in False
            do
                for is_timeseries in True
                do
                    python train_regression_model.py\
                        --n_runs 30\
                        --base_model $model\
                        --behavior_policy $pi_b\
                        --campaign $camp\
                        --is_mrdr $is_mrdr\
                        --is_timeseries_split $is_timeseries
                done
            done
        done
    done
done
```


## Evaluating Off-Policy Estimators

Here, we evaluate the estimation performances of the following OPE estimators:

- Direct Method (DM)
- Inverse Probability Weighting (IPW)
- Self-Normalized Inverse Probability Weighting (SNIPW)
- Doubly Robust (DR)
- Self-Normalized Doubly Robust (SNDR)
- Switch Inverse Probability Weighting (Switch-IPW)
- Switch Doubly Robust (Switch-DR)
- Doubly Robust with Optimistic Shrinkage (DRos)
-  More Robust Doubly Robust (MRDR)

For Switch-IPW, Switch-DR, and DRos, we use some different values of hyperparameters.

[benchmark_off_policy_estimators.py](https://github.com/st-tech/zr-obp/blob/master/benchmark/ope/benchmark_off_policy_estimators.py) implements the evaluation and comparison of OPE estimators using the open bandit dataset.
Note that you have to finish training a regression model (see the above section) before conducting the evaluation of OPE in the corresponding setting.
The detailed experimental procedures and results can be found in Section 5 of https://arxiv.org/abs/2008.07146

```
# run evaluation of OPE estimators with the full open bandit data
python benchmark_off_policy_estimators.py\
    --n_runs $n_runs\
    --base_model $base_model\ "logistic_regression" or "lightgbm"
    --behavior_policy $behavior_policy\ "random" or "bts"
    --campaign $campaign\ # "men", "women", or "all"
    --n_sim_to_compute_action_dist $n_sim_to_compute_action_dist\
    --test_size $test_size\
    --is_timeseries_split\ # in-sample or out-sample
    --n_jobs $n_jobs\
    --random_state $random_state
```
where
- `$n_runs` specifies the number of simulation runs with different bootstrap samples in the experiment to estimate standard deviations of the performance of OPE estimators (i.e., relative estimation error).
- $base_model_for_evaluation_policy specifies the base ML model for defining the regression model and should be one of "logistic_regression" or "lightgbm".
- `$campaign` specifies the campaign and should be one of 'all', 'men', or 'women'.
- `$n_sim_to_compute_action_dist` is the number of monte carlo simulation to compute the action distribution of a given evaluation policy.
- `$test_size` specifies the proportion of the dataset to include in the test split (when `$is_timeseries_split` is applied.)
- `$is_timeseries_split` is whether the data is split based on timestamp or not. If true, the out-sample performance of OPE is tested. See the relevant paper for details.
- `$n_jobs` is the maximum number of concurrently running jobs.

For example, the following command compares the estimation performances of the listed OPE estimators using Bernoulli TS as evaluation policy and Random as behavior policy in "All" campaign in the out-sample situation.

```bash
python benchmark_off_policy_estimators.py\
    --n_runs 10\
    --base_model logistic_regression\
    --behavior_policy random\
    --campaign all\
    --test_size 0.3\
    --is_timeseries_split True
```


```
for model in logistic_regression
do
    for pi_b in bts
    do
        for camp in all
        do
            for is_timeseries in True
            do
                python benchmark_off_policy_estimators.py\
                    --n_runs 30\
                    --base_model $model\
                    --behavior_policy $pi_b\
                    --campaign $camp\
                    --n_jobs 1\
                    --is_timeseries_split $is_timeseries
            done
        done
    done
done
```

## Results

We report the results of the benchmark experiments on the three campaigns (all, men, women) in the following tables.
We describe **Random -> Bernoulli TS** to represent the OPE situation where we use Bernoulli TS as a hypothetical evaluation policy and Random as a hypothetical behavior policy.
In contrast, we use **Bernoulli TS -> Random** to represent the situation where we use Random as a hypothetical evaluation policy and Bernoulli TS as a hypothetical behavior policy.
