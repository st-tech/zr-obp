# Examples with OBD
Here, we use the dataset and pipeline to implement and evaluate OPE.
We first evaluate well-known off-policy estimators with the ground-truth performance of a counterfactual policy.
We then select and use such an estimator to improve the platform’s fashion item recommendation.

## Additional Implementations

- [`dataset.py`](./dataset.py):
    We implement two ways of engeering original features.
    This includes **CONTEXT SET 1** (user features only, such as age, gender, and length of  membership) and **CONTEXT SET 2** (Context Set 1 plus user-item affinity induced by the number of past clicks observed between each user-item pair)

- [`logistic_bandit.py`](./logistic_bandit.py):
    We implement three logistic contextual bandit algorithms *logistic epsilon-greedy (logistic EGreedy)*, *logistic upper confidence bound (logistic UCB)*, and *logistic thompson sampling (logistic TS)*.
    They construct our counterfactual policy search space.
    Note that we modify the contextual bandit algorithms to adjust to ZOZOTOWN top-3 recommendation setting.
    For example, modified logistic TS selects three actions with the three highest sampled rewards.

## Running experiments

**Step 1. Evaluating a regression model**

We evaluate LightGBM as a regression model to be used in DM and DR.

```bash
python eval_regression_model.py\
    --n_splits 1\ # the number of train-test splits
    --behavior_policy bts\ # a behavior policy
    --campaign men # a campaign

# click prediction performance of a regression model
# =========================
# AUC: 0.615171
# RCE: 0.010273
# =========================
```

**Step 2. Selecting Off-policy Estimator** (Section 5.1)

We select the best off-policy estimator among DM, IPW, and DR.

```bash
python off_policy_estimator_selection.py\
    --n_splits 1\
    --n_estimators 10\ # the number of bagged samples used in bagging aggregation
    --behavior_policy bts\
    --campaign men

# relative estiamtion erros of three OPE estimators
# =========================
# random_state=0
# -----
# DM: 0.342532
# IPW: 0.684355
# DR: 1.035945
# =========================
```


**Step 3. Selecting Counterfactual Policy** (Section 5.2)

We evaluate the performance of counterfactual policies based on logistic contextual bandit with OPE estimators.

```bash
python cf_policy_selection.py\
    --n_splits 1\
    --n_estimators 10\
    --context_set 1\ # a used context set
    --counterfactual_policy logistic_ucb\ # a used counterfactual bandit algorithm
    --epsilon 0.1\ # an exploration hyperparameter
    --behavior_policy bts\
    --campaign men

# estimated policy values relative to the behavior policy (the Bernoulli TS here) of a counterfactual policy (the Logistic UCB with Context Set 1 here) by three OPE estimators
# =========================
# random_state=0
# -----
# DM: 0.826714
# IPW: 0.922185
# DR: 1.013117
# =========================
```


## References
1. Alina Beygelzimer and John Langford. The offset tree for learning with partial labels. In *Proceedings of the 15th ACM SIGKDD international conference on Knowledge discovery and data mining*, pages 129–138, 2009.

1. Olivier Chapelle and Lihong Li. An empirical evaluation of thompson sampling. In *Advances in neural information processing systems*, pages 2249–2257, 2011.

1. Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li. Doubly Robust Policy Evaluation and Optimization. *Statistical Science*, 29:485–511, 2014.

1. Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu. Lightgbm: A highly efficient gradient boosting decision tree. In *Advances in Neural Information Processing Systems*, pages 3146–3154, 2017.

1. Lihong Li, Wei Chu, John Langford, and Robert E Schapire. A Contextual-bandit Approach to Personalized News Article Recommendation. In *Proceedings of the 19th International Conference on World Wide Web*, pages 661–670. ACM, 2010.

1. Alex Strehl, John Langford, Lihong Li, and Sham M Kakade. Learning from Logged Implicit Exploration Data. In Advances in *Neural Information Processing Systems*, pages 2217–2225, 2010.
