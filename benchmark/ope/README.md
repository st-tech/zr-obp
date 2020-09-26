


### train regression model

```
for model in logistic_regression
do
    for pi_b in bts
    do
        for camp in men women all
        do
            screen python train_regression_model.py\
                --n_boot_samples 5\
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
for model in logistic_regression
do
    for pi_b in random
    do
        for camp in all men women
        do
            screen python benchmark_off_policy_estimators.py\
                --n_boot_samples 5\
                --base_model $model\
                --behavior_policy $pi_b\
                --campaign $camp
        done
    done
done
```

```
python benchmark_off_policy_estimators.py --n_boot_samples 3 --base_model lightgbm --behavior_policy random --campaign men --is_timeseries_split
```
