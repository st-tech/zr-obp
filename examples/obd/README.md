# Examples with OBD
Here, we use the open bandit dataset and pipeline to implement and evaluate OPE.
We first evaluate well-known off-policy estimators with the ground-truth performance of a counterfactual policy.
We then select and use such an estimator to improve the platform’s fashion item recommendation.

## Additional Implementations

- [`custom_dataset.py`](./custom_dataset.py):
    We implement two ways of engineering original features.
    This includes **CONTEXT SET 1** (user features only, such as age, gender, and length of  membership) and **CONTEXT SET 2** (Context Set 1 plus user-item affinity induced by the number of past clicks observed between each user-item pair).

## Running experiments

**Example Experimet 1. Evaluating Off-Policy Estimators**

We select the best off-policy estimator among Direct Method (DM), Inverse Probability Weighting (IPW), and Doubly Robust (DR).
[`./evaluate_off_policy_estimators.py`](./evaluate_off_policy_estimators.py) implements the evaluation of OPE estimators.

```bash
# run evaluation of OPE estimators.
python evaluate_off_policy_estimators.py\
    --n_boot_samples $n_boot_samples\
    --counterfactual_policy $counterfactual_policy\
    --behavior_policy $behavior_policy\
    --campaign $campaign\
    --random_state $random_state
```
where `$n_boot_samples` specifies the number of bootstrap samples to estimate confidence intervals of the performance of OPE estimators.
`$counterfactual_policy` and `$behavior_policy` specify the counterfactual and behavior policies, respectively.
They should be either 'bts' or 'random'.
`$campaign` specifies the campaign and should be one of 'all', 'men', or 'women'.

For example, the following command compares the estimation performance of the three OPE estimators by using Bernoulli TS as counterfactual policy and Random as behavior policy in "All" campaign.

```bash
python evaluate_off_policy_estimators.py\
    --n_boot_samples 10\
    --counterfactual_policy bts\
    --behavior_policy random\
    --campaign all

# relative estimation errors and their 95% confidence intervals of OPE estimators.
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
```


**Example Experimet 2. Evaluating Counterfactual Bandit Policy**

We evaluate the performance of counterfactual policies based on logistic contextual bandit algorithms in `obp.policy` module with OPE estimators in `obp.ope` module.
[`./evaluate_counterfactual_policy.py`](./evaluate_counterfactual_policy.py) implements the evaluation of the performance of counterfactual logistic bandit policies with the use of OPE estimators.

```bash
# run evaluation of a counterfacutal logistic bandit policy.
python evaluate_counterfactual_policy.py\
    --context_set $context_set\
    --counterfactual_policy $counterfactual_policy\
    --epsilon $epsilon\
    --behavior_policy $behavior_policy\
    --campaign $campaign\
    --random_state $random_state
```
where `$context_set` specifies the feature engineering method and should be either '1' or '2'.
`$counterfactual_policy` specifies the counterfactual logistic bandit policy and should be one of 'logistic_egreedy', 'logistic_ts', and 'logistic_ucb'.
`$epsilon` specifies the value of exploration hyperparameter and should be between 0 and 1.
`$behavior_policy` specifies the behavior policy and should be either 'bts' or 'random'.
`$campaign` specifies the campaign and should be one of 'all', 'men', or 'women'.


For example, the following command evaluates the performance of logistic_ucb policy (context_set='1' and exploration hyperparameter=`0.1`) by using the three OPE estimators and Random as behavior policy in "All" campaign.

```bash
python evaluate_counterfactual_policy.py\
    --context_set 1\
    --counterfactual_policy logistic_ucb\
    --epsilon 0.1\
    --behavior_policy random\
    --campaign all

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
```

