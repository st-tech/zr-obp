


### train regression model

```
for model in logistic_regression lightgbm
do
    for pi_b in random
    do
        for camp in men women all
        do
            screen python train_regression_model.py\
                --n_boot_samples 10\
                --base_model $model\
                --behavior_policy $pi_b\
                --campaign $camp
        done
    done
done
```

```
python train_regression_model.py --n_boot_samples 3 --base_model lightgbm --behavior_policy random --campaign men --is_timeseries_split
```


### benchmark off-policy estimators

```
for model in logistic_regression lightgbm
do
    for pi_b in bts random
    do
        for camp in men women all
        do
            benchmark_off_policy_estimators.py\
                --n_boot_samples 10\
                --base_model $model\
                --behavior_policy $pi_b\
                --campaign $camp\
                --is_timeseries_split
        done
    done
done
```

```
python benchmark_off_policy_estimators.py --n_boot_samples 3 --base_model lightgbm --behavior_policy random --campaign men --is_timeseries_split
```


### run cf policy search

```
for model in logistic_regression lightgbm random_forest
do
    for context in 1 2
    do
        for camp in men women all
        do
            screen python run_cf_policy_search.py\
                --context_set $context\
                --base_model $model\
                --behavior_policy bts\
                --campaign $camp
        done
    done
done
```

```
python run_cf_policy_search.py --context_set 1 --base_model logistic_regression --campaign men
```
