# Examples with OBD
Here, we use the dataset and pipeline to implement and evaluate OPE.
We first evaluate well-known off-policy estimators with the ground-truth performance of a counterfactual policy.
We then select and use such an estimator to improve the platform’s fashion item recommendation.

## Additional Implementations

- [`dataset.py`](./dataset.py):
    We implement two ways of engeering original features.
    This includes **CONTEXT SET 1** (user features only, such as age, gender, and length of  membership) and **CONTEXT SET 2** (Context Set 1 plus user-item affinity induced by the number of past clicks observed between each user-item pair).

## Running experiments

**Example 1. Evaluating Off-policy Estimator** (Section 5.1)

We select the best off-policy estimator among Direct Method, Inverse Probability Weighting, and Doubly Robust.

```bash
python evaluate_off_policy_estimators.py\
    --n_splits 5\
    --counterfactual_policy random\
    --behavior_policy bts\
    --campaign men

# relative estiamtion erros and their 95% confidence intervals of OPE estimators
# ==================================================
# random_state=12345
# --------------------------------------------------
#          mean  95.0% CI (lower)  95.0% CI (upper)
# dm   0.092205           0.05880           0.13488
# ipw  1.055920           0.83860           1.35543
# dr   0.928787           0.66004           1.18238
# ==================================================
```


**Example 2. Evaluating Counterfactual Bandit Policy** (Section 5.2)

We evaluate the performance of counterfactual policies based on logistic contextual bandit with OPE estimators.

```bash
python evaluate_counterfactual_policy.py\
    --context_set 1\ # a used context set for a counterfactual policy
    --counterfactual_policy logistic_ucb\ # a used counterfactual bandit algorithm
    --epsilon 0.1\ # an exploration hyperparameter
    --behavior_policy bts\
    --campaign men

# estimated policy values relative to the behavior policy (the Bernoulli TS here) of a counterfactual policy (the logistic UCB with Context Set 1 here) by three OPE estimators (IPW: inverse probability weighting, DM; Direct Method, DR: Doubly Robust)
# ======================================================================
# random_state=12345: counterfactual policy=logistic_ucb_0.0_1
# ----------------------------------------------------------------------
#      estimated_policy_value  relative_estimated_policy_value
# ipw                0.007184                         1.134279
# dm                 0.006955                         1.098153
# dr                 0.007447                         1.175799
# ======================================================================
```

