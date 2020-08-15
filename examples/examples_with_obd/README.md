# Examples with Open Bandit Dataset (OBD)
Here, we use the open bandit dataset and pipeline to implement and evaluate OPE.
We first evaluate well-known the estimation performances of off-policy estimators using the ground-truth policy value of a counterfactual policy.
We then evaluate the performances of some contextual bandit policies by using OPE to improve the platform’s fashion item recommendation.

## Descriptions

- `conf/`
  - [`./conf/batch_size_bts.yaml`]:
  The batch sizes used in the Bernoulli Thompson Sampling policy when running it on the ZOZOTOWN platform
  - [`./conf/prior_bts.yaml`]
  The prior hyperparameters used in the Bernoulli Thompson Sampling policy when running it on the ZOZOTOWN platform
  - [`./conf/lightgbm.yaml`]
  The hyperparameters of the LightGBM model that is used as the regression model in model dependent OPE estimators such as DM and DR

- [`custom_dataset.py`](./custom_dataset.py):
    We implement two ways of engineering original features.
    This includes **CONTEXT SET 1** (user features only, such as age, gender, and length of membership) and **CONTEXT SET 2** (Context Set 1 plus user-item affinity induced by the number of past clicks observed between each user-item pair).

## Running experiments

**Evaluating Off-Policy Estimators**

We evaluate the estimation performances of off-policy estimators, including Direct Method (DM), Inverse Probability Weighting (IPW), and Doubly Robust (DR).
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
where `$n_boot_samples` specifies the number of bootstrap samples to estimate confidence intervals of the performance of OPE estimators (*relative estimation error*).
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
# our evaluation of OPE procedure suggests that DM performs best among the three OPE estimators, because it has low variance property.
# (Note that this result is with the small sample data, and please use the full size data for a more reasonable experiment)
# ==================================================
# random_state=12345
# --------------------------------------------------
#          mean  95.0% CI (lower)  95.0% CI (upper)
# dm   0.213823          0.141678          0.277700
# ipw  1.158730          0.961905          1.533333
# dr   1.105379          0.901894          1.425447
# ==================================================
```

Please visit [Examples with Synthetic Data](https://github.com/st-tech/zr-obp/tree/master/examples/synthetic) to try the evaluation of OPE estimators with a larger dataset.


**Evaluating Counterfactual Bandit Policy**

We evaluate the performance of counterfactual policies based on contextual bandit algorithms in the policy module with OPE estimators in the ope module.
[`./evaluate_counterfactual_policy.py`](./evaluate_counterfactual_policy.py) implements the evaluation of the performance of counterfactual bandit policies with the use of OPE estimators.

```bash
# run evaluation of a counterfactual contextual bandit policy.
python evaluate_counterfactual_policy.py\
    --context_set $context_set\
    --counterfactual_policy $counterfactual_policy\
    --epsilon $epsilon\
    --behavior_policy $behavior_policy\
    --campaign $campaign\
    --random_state $random_state
```
where `$context_set` specifies the feature engineering method and should be either '1' or '2'.
`$counterfactual_policy` specifies the counterfactual contextual bandit policy and should be one of 'linear_egreedy', 'linear_ts', 'linear_ucb', 'logistic_egreedy', 'logistic_ts', and 'logistic_ucb'.
`$epsilon` specifies the value of exploration hyperparameter and must be between 0 and 1.
`$behavior_policy` specifies the behavior policy and should be either 'bts' or 'random'.
`$campaign` specifies the campaign and should be one of 'all', 'men', or 'women'.


For example, the following command evaluates the performance of Linear Epsilon Greedy (linear_egreedy) policy (context_set='1' and exploration hyperparameter=`0.1`) by using the three OPE estimators and Random as behavior policy in "Women" campaign.

```bash
python evaluate_counterfactual_policy.py\
    --context_set 1\
    --counterfactual_policy linear_egreedy\
    --epsilon 0.1\
    --behavior_policy random\
    --campaign women

# estimated policy values relative to the behavior policy (Random) of a counterfactual policy (linear epsilon greedy with Context Set 1)
# by the three OPE estimators (IPW: inverse probability weighting, DM; Direct Method, DR; Doubly Robust)
# in this example, DM predicts that the counterfactual policy outperforms the behavior policy by about 5.49%
# (Note that this result is with the small sample data, and please use the full size data for a more reasonable experiment)
# ======================================================================
# random_state=12345: counterfactual policy=linear_epsilon_greedy_0.1_1
# ----------------------------------------------------------------------
#      estimated_policy_value  relative_estimated_policy_value
# ipw                0.004600                         1.000000
# dm                 0.004853                         1.054967
# dr                 0.004642                         1.009075
# ======================================================================
```

