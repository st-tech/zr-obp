<p align="center">
  <img width="60%" src="./images/logo.png" />
</p>

# Open Bandit Pipeline: a research framework for bandit algorithms and off-policy evaluation

**[Docs](https://zr-obp.readthedocs.io/en/latest/)** | **[Google Group](https://groups.google.com/g/open-bandit-project)** | **[Installation](#installation)** | **[Usage](#usage)** | **[Slides](https://github.com/st-tech/zr-obp/tree/master/slides/slides_EN.pdf)** | **[Quickstart](https://github.com/st-tech/zr-obp/blob/master/examples/quickstart)** | **[Open Bandit Dataset](https://github.com/st-tech/zr-obp/tree/master/obd)** | **[日本語](https://github.com/st-tech/zr-obp/blob/master/README_JN.md)**

<details>
<summary><strong>Table of Contents</strong></summary>

- [Open Bandit Pipeline: a research framework for bandit algorithms and off-policy evaluation](#open-bandit-pipeline-a-research-framework-for-bandit-algorithms-and-off-policy-evaluation)
- [Overview](#overview)
  - [Open Bandit Dataset (OBD)](#open-bandit-dataset-obd)
  - [Open Bandit Pipeline (OBP)](#open-bandit-pipeline-obp)
    - [Algorithms and OPE Estimators Supported](#algorithms-and-ope-estimators-supported)
  - [Topics and Tasks](#topics-and-tasks)
- [Installation](#installation)
  - [Requirements](#requirements)
- [Usage](#usage)
  - [(1) Data loading and preprocessing](#1-data-loading-and-preprocessing)
  - [(2) Off-Policy Learning](#2-off-policy-learning)
  - [(3) Off-Policy Evaluation](#3-off-policy-evaluation)
- [Citation](#citation)
- [Project Team](#project-team)
- [Google Group](#google-group)
- [Contact](#contact)
- [License](#license)
- [References](#references)
  - [Papers](#papers)
  - [Projects](#projects)

</details>

# Overview

## Open Bandit Dataset (OBD)

*Open Bandit Dataset* is a public real-world logged bandit feedback data.
The dataset is provided by [ZOZO, Inc.](https://corp.zozo.com/en/about/profile/), the largest Japanese fashion e-commerce company with over 5 billion USD market capitalization (as of May 2020).
The company uses multi-armed bandit algorithms to recommend fashion items to users in a large-scale fashion e-commerce platform called [ZOZOTOWN](https://zozo.jp/).
The following figure presents examples of displayed fashion items as actions.

<p align="center">
  <img width="45%" src="./images/recommended_fashion_items.png" />
  <figcaption>
  <p align="center">
  Recommended fashion items as actions in ZOZOTOWN
  </p>
  </figcaption>
</p>

We collected the data in a 7-day experiment in late November 2019 on three “campaigns,” corresponding to all, men's, and women's items, respectively.
Each campaign randomly used either the Uniform Random algorithm or the Bernoulli Thompson Sampling (Bernoulli TS) algorithm, which was pre-trained for about a month before the data collection period.

<p align="center">
  <img width="70%" src="./images/statistics_of_obd.png" />
</p>

The small size version of our data is available at [./obd](https://github.com/st-tech/zr-obp/tree/master/obd).
This can be used for running [examples](https://github.com/st-tech/zr-obp/tree/master/examples).
We release the full size version of our data at [https://research.zozo.com/data.html](https://research.zozo.com/data.html).
Please download the full size version for research uses.
Please see [./obd/README.md](https://github.com/st-tech/zr-obp/blob/master/obd/README.md) for the description of the dataset.

## Open Bandit Pipeline (OBP)


*Open Bandit Pipeline* is a series of implementations of dataset preprocessing, OPE estimators, and the evaluation of OPE estimators.
This pipeline allows researchers to focus on building their own OPE estimator and easily compare it with others’ methods in realistic and reproducible ways.
Thus, it facilitates reproducible research on bandit algorithms and off-policy evaluation.

<p align="center">
  <img width="90%" src="./images/overview.png" />
  <figcaption>
  <p align="center">
    Structure of Open Bandit Pipeline
  </p>
  </figcaption>
</p>

Open Bandit Pipeline consists of the following main modules.

- **dataset module**: This module provides a data loader for Open Bandit Dataset and a flexible interface for handling logged bandit feedback. It also provides tools to generate synthetic bandit datasets.
- **policy module**: This module provides interfaces for online and offline bandit algorithms. It also implements several standard policy learning methods.
- **simulator module**: This module provides functions for conducting offline bandit simulation.
- **ope module**: This module provides interfaces for OPE estimators. It also implements several standard and advanced OPE estimators.

### Algorithms and OPE Estimators Supported

- Bandit Algorithms (implemented in **policy module**)
  - Online
    - Context-free
      - Random
      - Epsilon Greedy
      - Bernoulli Thompson Sampling
    - Contextual (Linear)
      - Linear Epsilon Greedy
      - [Linear Thompson Sampling](http://proceedings.mlr.press/v28/agrawal13)
      - [Linear Upper Confidence Bound](https://dl.acm.org/doi/pdf/10.1145/1772690.1772758)
    - Contextual (Logistic)
      - Logistic Epsilon Greedy
      - [Logistic Thompson Sampling](https://papers.nips.cc/paper/4321-an-empirical-evaluation-of-thompson-sampling)
      - [Logistic Upper Confidence Bound](https://dl.acm.org/doi/10.1145/2396761.2396767)
  - Offline (Off-Policy Learning)
    - [Inverse Probability Weighting (IPW) Learner](https://arxiv.org/abs/1503.02834)


- OPE Estimators (implemented in **ope module**)
  - [Replay Method (RM)](https://arxiv.org/abs/1003.5956)
  - [Direct Method (DM)](https://arxiv.org/abs/0812.4044)
  - [Inverse Probability Weighting (IPW)](https://scholarworks.umass.edu/cgi/viewcontent.cgi?article=1079&context=cs_faculty_pubs)
  - [Self-Normalized Inverse Probability Weighting (SNIPW)](https://papers.nips.cc/paper/5748-the-self-normalized-estimator-for-counterfactual-learning)
  - [Doubly Robust (DR)](https://arxiv.org/abs/1503.02834)
  - [Switch Estimators](https://arxiv.org/abs/1612.01205)
  - [More Robust Doubly Robust (MRDR)](https://arxiv.org/abs/1802.03493)
  - [Doubly Robust with Optimistic Shrinkage (DRos)](https://arxiv.org/abs/1907.09623)
  - [Double Machine Learning (DML)](https://arxiv.org/abs/2002.08536)

In addition to the above algorithms and estimators, the pipeline also provides flexible interfaces.
Therefore, researchers can easily implement their own algorithms or estimators and evaluate them with our data and pipeline.
Moreover, the pipeline provides an interface for handling real-world logged bandit feedback datasets.
Thus, practitioners can combine their own datasets with the pipeline and easily evaluate bandit algorithms' performances in their settings.


## Topics and Tasks
Currently, Open Bandit Dataset & Pipeline facilitate evaluation and comparison related to the following research topics.

- **Bandit Algorithms**: Our data include large-scale logged bandit feedback collected by the uniform random policy. Therefore, it enables the evaluation of new online bandit algorithms, including contextual and combinatorial algorithms, in a large real-world setting.


- **Off-Policy Evaluation**: We present implementations of behavior policies used when collecting datasets as a part of our pipeline. Our open data also contains logged bandit feedback data generated by *multiple* different bandit policies. Therefore, it enables the evaluation of off-policy evaluation with ground-truth for the performance of evaluation policies.


# Installation

You can install OBP using Python's package manager `pip`.

```
pip install obp
```

You can install OBP from source.
```bash
git clone https://github.com/st-tech/zr-obp
cd zr-obp
python setup.py install
```


## Requirements
- **python>=3.7.0**
- matplotlib>=3.2.2
- numpy>=1.18.1
- pandas>=0.25.1
- pyyaml>=5.1
- seaborn>=0.10.1
- scikit-learn>=0.23.1
- scipy>=1.4.1
- tqdm>=4.41.1


# Usage

We show an example of conducting offline evaluation of the performance of Bernoulli Thompson Sampling (BernoulliTS) as an evaluation policy using the *Inverse Probability Weighting (IPW)* and logged bandit feedback generated by the Random policy (behavior policy).
We see that only ten lines of code are sufficient to complete OPE from scratch.

```python
# a case for implementing OPE of the BernoulliTS policy using log data generated by the Random policy
from obp.dataset import OpenBanditDataset
from obp.policy import BernoulliTS
from obp.ope import OffPolicyEvaluation, InverseProbabilityWeighting as IPW

# (1) Data loading and preprocessing
dataset = OpenBanditDataset(behavior_policy='random', campaign='all')
bandit_feedback = dataset.obtain_batch_bandit_feedback()

# (2) Off-Policy Learning
evaluation_policy = BernoulliTS(
    n_actions=dataset.n_actions,
    len_list=dataset.len_list,
    is_zozotown_prior=True, # replicate the policy in the ZOZOTOWN production
    campaign="all",
    random_state=12345
)
action_dist = evaluation_policy.compute_batch_action_dist(
    n_sim=100000, n_rounds=bandit_feedback["n_rounds"]
)

# (3) Off-Policy Evaluation
ope = OffPolicyEvaluation(bandit_feedback=bandit_feedback, ope_estimators=[IPW()])
estimated_policy_value = ope.estimate_policy_values(action_dist=action_dist)

# estimated performance of BernoulliTS relative to the ground-truth performance of Random
relative_policy_value_of_bernoulli_ts = estimated_policy_value['ipw'] / bandit_feedback['reward'].mean()
print(relative_policy_value_of_bernoulli_ts)
1.198126...
```

A formal introduction with the same example can be found at [quickstart](https://github.com/st-tech/zr-obp/tree/master/examples/quickstart).
Below, we explain some important features in the example.


## (1) Data loading and preprocessing

We prepare an easy-to-use data loader for Open Bandit Dataset.

```python
# load and preprocess raw data in "ALL" campaign collected by the Random policy
dataset = OpenBanditDataset(behavior_policy='random', campaign='all')
# obtain logged bandit feedback generated by the behavior policy
bandit_feedback = dataset.obtain_batch_bandit_feedback()

print(bandit_feedback.keys())
dict_keys(['n_rounds', 'n_actions', 'action', 'position', 'reward', 'pscore', 'context', 'action_context'])
```

Users can implement their own feature engineering in the `pre_process` method of `obp.dataset.OpenBanditDataset` class.
We show an example of implementing some new feature engineering processes in [`./examples/examples_with_obd/custom_dataset.py`](https://github.com/st-tech/zr-obp/blob/master/benchmark/cf_policy_search/custom_dataset.py).

Moreover, by following the interface of `obp.dataset.BaseBanditDataset` class, one can handle future open datasets for bandit algorithms other than our Open Bandit Dataset.
`dataset` module also provide a class to generate synthetic bandit datasets.

## (2) Off-Policy Learning

After preparing a dataset, we now compute the action choice probability of BernoulliTS in the ZOZOTOWN production.
Then, we can use it as the evaluation policy.

```python
# define evaluation policy (the Bernoulli TS policy here)
# by activating the `is_zozotown_prior` argument of BernoulliTS, we can replicate BernoulliTS used in ZOZOTOWN production.
evaluation_policy = BernoulliTS(
    n_actions=dataset.n_actions,
    len_list=dataset.len_list,
    is_zozotown_prior=True, # replicate the BernoulliTS policy in the ZOZOTOWN production
    campaign="all",
    random_state=12345
)
# compute the distribution over actions by the evaluation policy using Monte Carlo simulation
# action_dist is an array of shape (n_rounds, n_actions, len_list)
# representing the distribution over actions made by the evaluation policy
action_dist = evaluation_policy.compute_batch_action_dist(
    n_sim=100000, n_rounds=bandit_feedback["n_rounds"]
)
```

When `is_zozotown_prior=False`, non-informative prior distribution is used.
The `compute_batch_action_dist` method of `BernoulliTS` computes the action choice probabilities based on given hyperparameters of the beta distribution. `action_dist` is an array representing the distribution over actions made by the evaluation policy.

Users can implement their own bandit algorithms by following the interfaces implemented in [`./obp/policy/base.py`](https://github.com/st-tech/zr-obp/blob/master/obp/policy/base.py).

## (3) Off-Policy Evaluation

Our final step is **off-policy evaluation** (OPE), which attempts to estimate the performance of a new decision making policy using log data generated by a behavior, past policy.
Our pipeline also provides an easy procedure for doing OPE as follows.

```python
# estimate the policy value of BernoulliTS based on the distribution over actions by that policy
# it is possible to set multiple OPE estimators to the `ope_estimators` argument
ope = OffPolicyEvaluation(bandit_feedback=bandit_feedback, ope_estimators=[IPW()])
estimated_policy_value = ope.estimate_policy_values(action_dist=action_dist)
print(estimated_policy_value)
{'ipw': 0.004553...} # dictionary containing estimated policy values by each OPE estimator.

# compare the estimated performance of BernoulliTS (evaluation policy)
# with the ground-truth performance of Random (behavior policy)
relative_policy_value_of_bernoulli_ts = estimated_policy_value['ipw'] / bandit_feedback['reward'].mean()
# our OPE procedure suggests that BernoulliTS improves Random by 19.81%
print(relative_policy_value_of_bernoulli_ts)
1.198126...
```
Users can implement their own OPE estimator by following the interface of `obp.ope.BaseOffPolicyEstimator` class. `obp.ope.OffPolicyEvaluation` class summarizes and compares the estimated policy values by several off-policy estimators.
A detailed usage of this class can be found at [quickstart](https://github.com/st-tech/zr-obp/tree/master/examples/quickstart). `bandit_feedback['reward'].mean()` is the empirical mean of factual rewards (on-policy estimate of the policy value) in the log and thus is the ground-truth performance of the behavior policy (the Random policy in this example.).


# Citation
If you use our dataset and pipeline in your work, please cite our paper:

Yuta Saito, Shunsuke Aihara, Megumi Matsutani, Yusuke Narita.
**Large-scale Open Dataset, Pipeline, and Benchmark for Bandit Algorithms** [https://arxiv.org/abs/2008.07146](https://arxiv.org/abs/2008.07146)

Bibtex:
```
@article{saito2020large,
  title={Large-scale Open Dataset, Pipeline, and Benchmark for Bandit Algorithms},
  author={Saito, Yuta and Shunsuke Aihara and Megumi Matsutani and Yusuke Narita},
  journal={arXiv preprint arXiv:2008.07146},
  year={2020}
}
```


# Project Team

- [Yuta Saito](https://usaito.github.io/) (**Main Contributor**; Hanjuku-kaso Co., Ltd. / Tokyo Institute of Technology)
- [Shunsuke Aihara](https://www.linkedin.com/in/shunsukeaihara/) (ZOZO Technologies, Inc.)
- Megumi Matsutani (ZOZO Technologies, Inc.)
- [Yusuke Narita](https://www.yusuke-narita.com/) (Hanjuku-kaso Co., Ltd. / Yale University)

# Google Group
If you are interested in the Open Bandit Project, we can follow the updates at its google group: https://groups.google.com/g/open-bandit-project

# Contact
For any question about the paper, data, and pipeline, feel free to contact: saito@hanjuku-kaso.com

# License
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

# References

## Papers
1. Alina Beygelzimer and John Langford. [The offset tree for learning with partial labels](https://arxiv.org/abs/0812.4044). In
*Proceedings of the 15th ACM SIGKDD international conference on Knowledge discovery and data mining*, pages 129–138, 2009.

2. Olivier Chapelle and Lihong Li. [An empirical evaluation of thompson sampling](https://papers.nips.cc/paper/4321-an-empirical-evaluation-of-thompson-sampling). In *Advances in neural information processing systems*, pages 2249–2257, 2011.

3. Lihong Li, Wei Chu, John Langford, and Xuanhui Wang. [Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms](https://arxiv.org/abs/1003.5956). In *Proceedings of the Fourth ACM International Conference on Web Search and Data Mining*, pages 297–306, 2011.

4. Alex Strehl, John Langford, Lihong Li, and Sham M Kakade. [Learning from Logged Implicit Exploration Data](https://arxiv.org/abs/1003.0120). In *Advances in Neural Information Processing Systems*, pages 2217–2225, 2010.

5.  Doina Precup, Richard S. Sutton, and Satinder Singh. [Eligibility Traces for Off-Policy Policy Evaluation](https://scholarworks.umass.edu/cgi/viewcontent.cgi?article=1079&context=cs_faculty_pubs). In *Proceedings of the 17th International Conference on Machine Learning*, 759–766. 2000.

6.  Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li. [Doubly Robust Policy Evaluation and Optimization](https://arxiv.org/abs/1503.02834). *Statistical Science*, 29:485–511, 2014.

7. Adith Swaminathan and Thorsten Joachims. [The Self-normalized Estimator for Counterfactual Learning](https://papers.nips.cc/paper/5748-the-self-normalized-estimator-for-counterfactual-learning). In *Advances in Neural Information Processing Systems*, pages 3231–3239, 2015.

8. Dhruv Kumar Mahajan, Rajeev Rastogi, Charu Tiwari, and Adway Mitra. [LogUCB: An Explore-Exploit Algorithm for Comments Recommendation](https://dl.acm.org/doi/10.1145/2396761.2396767). In *Proceedings of the 21st ACM international conference on Information and knowledge management*, 6–15. 2012.

9.  Lihong Li, Wei Chu, John Langford, Taesup Moon, and Xuanhui Wang. [An Unbiased Offline Evaluation of Contextual Bandit Algorithms with Generalized Linear Models](http://proceedings.mlr.press/v26/li12a.html). In *Journal of Machine Learning Research: Workshop and Conference Proceedings*, volume 26, 19–36. 2012.

10. Yu-Xiang Wang, Alekh Agarwal, and Miroslav Dudik. [Optimal and Adaptive Off-policy Evaluation in Contextual Bandits](https://arxiv.org/abs/1612.01205). In *Proceedings of the 34th International Conference on Machine Learning*, 3589–3597. 2017.

11. Mehrdad Farajtabar, Yinlam Chow, and Mohammad Ghavamzadeh. [More Robust Doubly Robust Off-policy Evaluation](https://arxiv.org/abs/1802.03493). In *Proceedings of the 35th International Conference on Machine Learning*, 1447–1456. 2018.

12. Nathan Kallus and Masatoshi Uehara. [Intrinsically Efficient, Stable, and Bounded Off-Policy Evaluation for Reinforcement Learning](https://arxiv.org/abs/1906.03735). In *Advances in Neural Information Processing Systems*. 2019.

13. Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudík. [Doubly Robust Off-policy Evaluation with Shrinkage](https://arxiv.org/abs/1907.09623). In *Proceedings of the 37th International Conference on Machine Learning*, 2020.

14.  Yusuke Narita, Shota Yasui, and Kohei Yata. [Off-policy Bandit and Reinforcement Learning](https://arxiv.org/abs/2002.08536). *arXiv preprint arXiv:2002.08536*, 2020.

15. Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, and Jure Leskovec. [Open Graph Benchmark: Datasets for Machine Learning on Graphs](https://arxiv.org/abs/2005.00687). *arXiv preprint arXiv:2005.00687*, 2020.


## Projects
This project is strongly inspired by **Open Graph Benchmark** --a collection of benchmark datasets, data loaders, and evaluators for graph machine learning:
[[github](https://github.com/snap-stanford/ogb)] [[project page](https://ogb.stanford.edu)] [[paper](https://arxiv.org/abs/2005.00687)].
