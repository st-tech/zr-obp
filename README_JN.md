<div align="center"><img src="https://raw.githubusercontent.com/st-tech/zr-obp/master/images/logo.png" width="60%"/></div>

[![pypi](https://img.shields.io/pypi/v/obp.svg)](https://pypi.python.org/pypi/obp)
[![Python](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue)](https://www.python.org)
[![Downloads](https://pepy.tech/badge/obp)](https://pepy.tech/project/obp)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/st-tech/zr-obp)
![GitHub last commit](https://img.shields.io/github/last-commit/st-tech/zr-obp)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/arXiv-2008.07146-b31b1b.svg)](https://arxiv.org/abs/2008.07146)

[[arXiv]](https://arxiv.org/abs/2008.07146)

# Open Bandit Pipeline: a research framework for bandit algorithms and off-policy evaluation

**[ドキュメント](https://zr-obp.readthedocs.io/en/latest/)** | **[Google Group](https://groups.google.com/g/open-bandit-project)** | **[チュートリアル](https://sites.google.com/cornell.edu/recsys2021tutorial)** | **[インストール](#インストール)** | **[使用方法](#使用方法)** | **[スライド](./slides/slides_JN.pdf)**  | **[Quickstart](./examples/quickstart)** | **[Open Bandit Dataset](./obd/README_JN.md)** | **[解説ブログ記事](https://techblog.zozo.com/entry/openbanditproject)**

<details>
<summary><strong>Table of Contents</strong></summary>

- [Open Bandit Pipeline: a research framework for bandit algorithms and off-policy evaluation](#open-bandit-pipeline-a-research-framework-for-bandit-algorithms-and-off-policy-evaluation)
- [概要](#概要)
  - [Open Bandit Dataset](#open-bandit-dataset)
  - [Open Bandit Pipeline](#open-bandit-pipeline)
    - [実装されているバンディットアルゴリズムとオフ方策推定量](#実装されているバンディットアルゴリズムとオフ方策推定量)
  - [トピックとタスク](#トピックとタスク)
- [インストール](#インストール)
  - [依存パッケージ](#依存パッケージ)
- [使用方法](#使用方法)
  - [(1) データの読み込みと前処理](#1-データの読み込みと前処理)
  - [(2) オフ方策学習](#2-オフ方策学習)
  - [(3) オフ方策評価 （Off-Policy Evaluation）](#3-オフ方策評価-off-policy-evaluation)
- [引用](#引用)
- [Google Group](#google-group)
- [ライセンス](#ライセンス)
- [プロジェクトチーム](#プロジェクトチーム)
- [連絡先](#連絡先)
- [参考](#参考)

</details>

# 概要

## Open Bandit Dataset

*Open Bandit Dataset*は, バンディットアルゴリズムやオフ方策評価にまつわる研究を促進するための大規模公開実データです.
本データセットは, 日本最大のファッションEコマース企業である[株式会社ZOZO](https://corp.zozo.com/about/profile/)が提供しています.
同社が運営する大規模ファッションECサイト[ZOZOTOWN](https://zozo.jp/)では, いくつかの多腕バンディットアルゴリズムを用いてユーザにファッションアイテムを推薦しています.
バンディットアルゴリズムによるファッションアイテム推薦の例は以下の図1の通りです.
各ユーザリクエストに対して, 3つのファッションアイテムが同時に推薦されることがわかります.

<div align="center"><img src="https://raw.githubusercontent.com/st-tech/zr-obp/master/images/recommended_fashion_items.png" width="45%"/></div>
<figcaption>
<p align="center">
図1. ZOZOTOWNにおけるファッションアイテムの推薦の例
</p>
</figcaption>


2019年11月下旬の7日間にわたるデータ収集実験において, 全アイテム(all)・男性用アイテム(men)・女性用アイテム(women)に対応する3つの「キャンペーン」でデータを収集しました.
それぞれのキャンペーンでは, 各ユーザのインプレッションに対してランダム方策(Random)またはトンプソン抽出方策(Bernoulli Thompson Sampling; Bernoulli TS)のいずれかを確率的にランダムに選択して適用しています.
図2はOpen Bandit Datasetの記述統計を示しています.

<div align="center"><img src="https://raw.githubusercontent.com/st-tech/zr-obp/master/images/obd_stats.png" width="90%"/></div>
<figcaption>
  <p align="center">
    図2. Open Bandit Datasetのキャンペーンとデータ収集方策ごとの記述統計
  </p>
</figcaption>


[実装例](./examples)を実行するための少量版データは, [./obd/](./obd)にあります.
Open Bandit Datasetのフルサイズ版は[https://research.zozo.com/data.html](https://research.zozo.com/data.html)にあります.
動作確認等には少量版を, 研究用途にはフルサイズ版を活用してください.

## Open Bandit Pipeline

*Open Bandit Pipeline*は, データセットの前処理・オフ方策学習・オフ方策推定量の評価を簡単に行うためのPythonパッケージです.
Open Bandit Pipelineを活用することで, 研究者はオフ方策推定量 (OPE estimator) の実装に集中して現実的で再現性のある方法で他の手法との性能比較を行うことができるようになります.
オフ方策評価(Off-Policy Evaluation)については, [こちらのブログ記事](https://techblog.zozo.com/entry/openbanditproject)をご確認ください.

<div align="center"><img src="https://raw.githubusercontent.com/st-tech/zr-obp/master/images/overview.png" width="80%"/></div>
<figcaption>
<p align="center">
  図3. Open Bandit Pipelineの構成
</p>
</figcaption>

Open Bandit Pipelineは, 以下の主要モジュールで構成されています.

- [**datasetモジュール**](./obp/dataset): このモジュールは, Open Bandit Dataset用のデータ読み込みクラスとデータの前処理するための柔軟なインターフェースを提供します. また人工データを生成するクラスや多クラス分類データをバンディットデータに変換するためのクラスも実装しています.
- [**policyモジュール**](./obp/policy): このモジュールは, バンディットアルゴリズムのためのインターフェイスを提供します. 加えて, いくつかの標準なバンディットアルゴリズムを実装しています.
- [**opeモジュール**](./obp/ope):　このモジュールは, いくつかの標準的なオフ方策推定量を実装しています. また新たにオフ方策推定量を実装するためのインターフェースも提供しています.


### 実装されているバンディットアルゴリズムとオフ方策推定量

<details>
<summary><strong>バンディットアルゴリズム (policy moduleに実装)</strong></summary>

- Online
  - Non-Contextual (Context-free)
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
  - Neural Network-based Policy Learner

</details>

<details>
<summary><strong>オフ方策推定量 (ope moduleに実装)</strong></summary>

- OPE of Online Bandit Algorithms
  - [Replay Method (RM)](https://arxiv.org/abs/1003.5956)
- OPE of Offline Bandit Algorithms
  - [Direct Method (DM)](https://arxiv.org/abs/0812.4044)
  - [Inverse Probability Weighting (IPW)](https://scholarworks.umass.edu/cgi/viewcontent.cgi?article=1079&context=cs_faculty_pubs)
  - [Self-Normalized Inverse Probability Weighting (SNIPW)](https://papers.nips.cc/paper/5748-the-self-normalized-estimator-for-counterfactual-learning)
  - [Doubly Robust (DR)](https://arxiv.org/abs/1503.02834)
  - [Switch Estimators](https://arxiv.org/abs/1612.01205)
  - [More Robust Doubly Robust (MRDR)](https://arxiv.org/abs/1802.03493)
  - [Doubly Robust with Optimistic Shrinkage (DRos)](https://arxiv.org/abs/1907.09623)
  - [Double Machine Learning (DML)](https://arxiv.org/abs/2002.08536)
- OPE of Offline Slate Bandit Algorithms
  - [Independent Inverse Propensity Scoring (IIPS)](https://arxiv.org/abs/1804.10488)
  - [Reward Interaction Inverse Propensity Scoring (RIPS)](https://arxiv.org/abs/2007)
- OPE of Offline Bandit Algorithms with Continuous Actions
  - [Kernelized Inverse Probability Weighting](https://arxiv.org/abs/1802.06037)
  - [Kernelized Self-Normalized Inverse Probability Weighting](https://arxiv.org/abs/1802.06037)
  - [Kernelized Doubly Robust](https://arxiv.org/abs/1802.06037)

</details>

Open Bandit Pipelineは, 上記のアルゴリズムやオフ方策推定量に加えて柔軟なインターフェースも提供しています.
したがって研究者は, 独自のバンディットアルゴリズムや推定量を容易に実装することでそれらの性能を評価できます.
さらにOpen Bandit Pipelineは, 実バンディットフィードバックデータを扱うためのインタフェースを含んでいます.
したがって, エンジニアやデータサイエンティストなどの実践者は, 自社のデータセットをOpen Bandit Pipelineと組み合わせることで簡単にオフ方策評価を行うことができます.

## トピックとタスク

Open Bandit Dataset及びOpen Bandit Pipelineでは, 以下の研究テーマに関する実験評価を行うことができます.

- **バンディットアルゴリズムの性能評価 (Evaluation of Bandit Algorithms)**：Open Bandit Datasetには, ランダム方策によって収集された大規模なログデータが含まれています. それを用いることで, 新しいオンラインバンディットアルゴリズムの性能を評価することが可能です.

- **オフ方策評価の正確さの評価 (Evaluation of Off-Policy Evaluation)**：Open Bandit Datasetは, 複数の方策を実システム上で同時に走らせることにより生成されたログデータで構成されています. またOpen Bandit Pipelineを用いることで, データ収集に用いられた方策を再現できます. そのため, オフ方策推定量の推定精度の評価を行うことができます.


# インストール

以下の通り, `pip`を用いてOpen Bandit Pipelineをダウンロードできます.

```bash
pip install obp
```

また, 本リポジトリをcloneしてセットアップすることもできます.

```bash
git clone https://github.com/st-tech/zr-obp
cd zr-obp
python setup.py install
```

Pythonおよび利用パッケージのバージョンは以下の通りです。

```
[tool.poetry.dependencies]
python = ">=3.7.1,<3.10"
torch = "^1.9.0"
scikit-learn = "^0.24.2"
pandas = "^1.3.2"
numpy = "^1.21.2"
matplotlib = "^3.4.3"
tqdm = "^4.62.2"
scipy = "^1.7.1"
PyYAML = "^5.4.1"
seaborn = "^0.11.2"
pyieoe = "^0.1.1"
pingouin = "^0.4.0"
```

これらのパッケージのバージョンが異なると、使用方法や挙動が本書執筆時点と異なる場合があるので、注意してください。

# 使用方法

ここでは, Open Bandit Pipelineの使用法を説明します. 具体例として, Open Bandit Datasetを用いて, トンプソン抽出方策の性能をオフライン評価する流れを実装します. 人工データや多クラス分類データを用いたオフ方策評価の実装法は, [英語版のREAMDE](https://github.com/st-tech/zr-obp/blob/master/README.md)や[examples/quickstart/](https://github.com/st-tech/zr-obp/tree/master/examples/quickstart)をご確認ください.

以下に示すように, 約10行のコードでオフ方策評価の流れを実装できます.

```python
# Inverse Probability Weightingとランダム方策によって生成されたログデータを用いて, BernoulliTSの性能をオフラインで評価する
from obp.dataset import OpenBanditDataset
from obp.policy import BernoulliTS
from obp.ope import OffPolicyEvaluation, InverseProbabilityWeighting as IPW

# (1) データの読み込みと前処理
dataset = OpenBanditDataset(behavior_policy='random', campaign='all')
bandit_feedback = dataset.obtain_batch_bandit_feedback()

# (2) オフ方策学習
evaluation_policy = BernoulliTS(
    n_actions=dataset.n_actions,
    len_list=dataset.len_list,
    is_zozotown_prior=True,
    campaign="all",
    random_state=12345
)
action_dist = evaluation_policy.compute_batch_action_dist(
    n_sim=100000, n_rounds=bandit_feedback["n_rounds"]
)

# (3) オフ方策評価
ope = OffPolicyEvaluation(bandit_feedback=bandit_feedback, ope_estimators=[IPW()])
estimated_policy_value = ope.estimate_policy_values(action_dist=action_dist)

# ランダム方策に対するトンプソン抽出方策の性能の改善率（相対クリック率）
relative_policy_value_of_bernoulli_ts = estimated_policy_value['ipw'] / bandit_feedback['reward'].mean()
print(relative_policy_value_of_bernoulli_ts)
1.198126...
```

以下, 重要な要素について説明します.

## (1) データの読み込みと前処理

Open Bandit Pipelineには, Open Bandit Dataset用のデータ読み込みインターフェースを用意しています.
これを用いることで, Open Bandit Datasetの読み込みや前処理を簡潔に行うことができます.

```python
# 「全アイテムキャンペーン (all)」においてランダム方策が集めたログデータを読み込む.
# OpenBanditDatasetクラスにはデータを収集した方策とキャンペーンを指定する.
dataset = OpenBanditDataset(behavior_policy='random', campaign='all')

# オフ方策学習やオフ方策評価に用いるログデータを得る.
bandit_feedback = dataset.obtain_batch_bandit_feedback()

print(bandit_feedback.keys())
# dict_keys(['n_rounds', 'n_actions', 'action', 'position', 'reward', 'pscore', 'context', 'action_context'])
```

`obp.dataset.OpenBanditDataset` クラスの `pre_process` メソッドに, 独自の特徴量エンジニアリングを実装することもできます. [`custom_dataset.py`](https://github.com/st-tech/zr-obp/blob/master/benchmark/cf_policy_search/custom_dataset.py)には, 新しい特徴量エンジニアリングを実装する例を示しています. また, `obp.dataset.BaseBanditDataset`クラスのインターフェースに従って新たなクラスを実装することで, 将来公開されるであろうOpen Bandit Dataset以外のバンディットデータセットや自社に特有のバンディットデータを扱うこともできます.

## (2) オフ方策学習

前処理の後は, 次のようにして**オフ方策学習**を実行します.

```python
# 評価対象のアルゴリズムを定義. ここでは, トンプソン抽出方策の性能をオフライン評価する.
# 研究者が独自に実装したバンディット方策を用いることもできる.
evaluation_policy = BernoulliTS(
    n_actions=dataset.n_actions,
    len_list=dataset.len_list,
    is_zozotown_prior=True, # ZOZOTOWN上での挙動を再現
    campaign="all",
    random_state=12345
)
# シミュレーションを用いて、トンプソン抽出方策による行動選択確率を算出.
action_dist = evaluation_policy.compute_batch_action_dist(
    n_sim=100000, n_rounds=bandit_feedback["n_rounds"]
)
```

`BernoulliTS`の`compute_batch_action_dist`メソッドは, 与えられたベータ分布のパラメータに基づいた行動選択確率(`action_dist`)をシミュレーションによって算出します. またユーザは[`./obp/policy/base.py`](https://github.com/st-tech/zr-obp/blob/master/obp/policy/base.py)に実装されているインターフェースに従うことで独自のバンディットアルゴリズムを実装し, その性能を評価することもできます.


## (3) オフ方策評価 （Off-Policy Evaluation）

最後のステップは, ログデータを用いてバンディットアルゴリズムの性能をオフライン評価する**オフ方策評価**です.
Open Bandit Pipelineを使うことで, 次のようにオフ方策評価を実装できます.

```python
# IPW推定量を用いてトンプソン抽出方策の性能をオフライン評価する.
# OffPolicyEvaluationクラスには, オフライン評価に用いるログバンディットデータと用いる推定量を渡す（複数設定可）.
ope = OffPolicyEvaluation(bandit_feedback=bandit_feedback, ope_estimators=[IPW()])
estimated_policy_value = ope.estimate_policy_values(action_dist=action_dist)
print(estimated_policy_value)
{'ipw': 0.004553...}　# 設定されたオフ方策推定量による性能の推定値を含んだ辞書.

# トンプソン抽出方策の性能の推定値とランダム方策の真の性能を比較する.
relative_policy_value_of_bernoulli_ts = estimated_policy_value['ipw'] / bandit_feedback['reward'].mean()
# オフ方策評価によって, トンプソン抽出方策の性能はランダム方策の性能を19.81%上回ると推定された.
print(relative_policy_value_of_bernoulli_ts)
1.198126...
```

`obp.ope.BaseOffPolicyEstimator` クラスのインターフェースに従うことで, 独自のオフ方策推定量を実装することもできます. これにより新たなオフ方策推定量の推定精度を検証することが可能です.
また, `obp.ope.OffPolicyEvaluation`の`ope_estimators`に複数のオフ方策推定量を設定することで, 複数の推定量による推定値を同時に得ることも可能です. `bandit_feedback['reward'].mean()` は観測された報酬の経験平均値（オン方策推定）であり, ランダム方策の真の性能を表します.


# 引用
Open Bandit DatasetやOpen Bandit Pipelineを活用して論文やブログ記事等を執筆された場合, 以下の論文を引用していただくようお願いいたします.

Yuta Saito, Shunsuke Aihara, Megumi Matsutani, Yusuke Narita.<br>
**Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation**<br>
[https://arxiv.org/abs/2008.07146](https://arxiv.org/abs/2008.07146)

Bibtex:
```
@article{saito2020open,
  title={Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation},
  author={Saito, Yuta and Shunsuke, Aihara and Megumi, Matsutani and Yusuke, Narita},
  journal={arXiv preprint arXiv:2008.07146},
  year={2020}
}
```

# Google Group
本プロジェクトに関する最新情報は次のGoogle Groupにて随時お知らせしています. ぜひご登録ください: https://groups.google.com/g/open-bandit-project

# コントリビューション
Open Bandit Pipelineへのどんな貢献も歓迎いたします. プロジェクトに貢献するためのガイドラインは, [CONTRIBUTING.md](./CONTRIBUTING.md)を参照してください。

# ライセンス
このプロジェクトはApache 2.0ライセンスを採用しています. 詳細は, [LICENSE](https://github.com/st-tech/zr-obp/blob/master/LICENSE)を参照してください.

# プロジェクトチーム

- [齋藤優太](https://usaito.github.io/) (**Main Contributor**; 半熟仮想株式会社 / コーネル大学)
- [粟飯原俊介](https://www.linkedin.com/in/shunsukeaihara/) (ZOZO研究所)
- 松谷恵 (ZOZO研究所)
- [成田悠輔](https://www.yusuke-narita.com/) (半熟仮想株式会社 / イェール大学)

## 開発メンバー
- [野村将寛](https://twitter.com/nomuramasahir0) (株式会社サイバーエージェント / 半熟仮想株式会社)
- [高山晃一](https://fullflu.hatenablog.com/) (半熟仮想株式会社)
- [黒岩稜](https://kurorororo.github.io) (トロント大学 / 半熟仮想株式会社)
- [清原明加](https://sites.google.com/view/harukakiyohara) (東京工業大学 / 半熟仮想株式会社)

# 連絡先
論文やOpen Bandit Dataset, Open Bandit Pipelineに関するご質問は, 次のメールアドレス宛にお願いいたします: saito@hanjuku-kaso.com

# 参考

<details>
<summary><strong>論文</strong></summary>

1. Alina Beygelzimer and John Langford. [The offset tree for learning with partial labels](https://arxiv.org/abs/0812.4044). In *Proceedings of the 15th ACM SIGKDD International Conference on Knowledge Discovery&Data Mining*, 129–138, 2009.

2. Olivier Chapelle and Lihong Li. [An empirical evaluation of thompson sampling](https://papers.nips.cc/paper/4321-an-empirical-evaluation-of-thompson-sampling). In *Advances in Neural Information Processing Systems*, 2249–2257, 2011.

3. Lihong Li, Wei Chu, John Langford, and Xuanhui Wang. [Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms](https://arxiv.org/abs/1003.5956). In *Proceedings of the Fourth ACM International Conference on Web Search and Data Mining*, 297–306, 2011.

4. Alex Strehl, John Langford, Lihong Li, and Sham M Kakade. [Learning from Logged Implicit Exploration Data](https://arxiv.org/abs/1003.0120). In *Advances in Neural Information Processing Systems*, 2217–2225, 2010.

5.  Doina Precup, Richard S. Sutton, and Satinder Singh. [Eligibility Traces for Off-Policy Policy Evaluation](https://scholarworks.umass.edu/cgi/viewcontent.cgi?article=1079&context=cs_faculty_pubs). In *Proceedings of the 17th International Conference on Machine Learning*, 759–766. 2000.

6.  Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li. [Doubly Robust Policy Evaluation and Optimization](https://arxiv.org/abs/1503.02834). *Statistical Science*, 29:485–511, 2014.

7. Adith Swaminathan and Thorsten Joachims. [The Self-normalized Estimator for Counterfactual Learning](https://papers.nips.cc/paper/5748-the-self-normalized-estimator-for-counterfactual-learning). In *Advances in Neural Information Processing Systems*, 3231–3239, 2015.

8. Dhruv Kumar Mahajan, Rajeev Rastogi, Charu Tiwari, and Adway Mitra. [LogUCB: An Explore-Exploit Algorithm for Comments Recommendation](https://dl.acm.org/doi/10.1145/2396761.2396767). In *Proceedings of the 21st ACM international conference on Information and knowledge management*, 6–15. 2012.

9.  Lihong Li, Wei Chu, John Langford, Taesup Moon, and Xuanhui Wang. [An Unbiased Offline Evaluation of Contextual Bandit Algorithms with Generalized Linear Models](http://proceedings.mlr.press/v26/li12a.html). In *Journal of Machine Learning Research: Workshop and Conference Proceedings*, volume 26, 19–36. 2012.

10. Yu-Xiang Wang, Alekh Agarwal, and Miroslav Dudik. [Optimal and Adaptive Off-policy Evaluation in Contextual Bandits](https://arxiv.org/abs/1612.01205). In *Proceedings of the 34th International Conference on Machine Learning*, 3589–3597. 2017.

11. Mehrdad Farajtabar, Yinlam Chow, and Mohammad Ghavamzadeh. [More Robust Doubly Robust Off-policy Evaluation](https://arxiv.org/abs/1802.03493). In *Proceedings of the 35th International Conference on Machine Learning*, 1447–1456. 2018.

12. Nathan Kallus and Masatoshi Uehara. [Intrinsically Efficient, Stable, and Bounded Off-Policy Evaluation for Reinforcement Learning](https://arxiv.org/abs/1906.03735). In *Advances in Neural Information Processing Systems*. 2019.

13. Yi Su, Lequn Wang, Michele Santacatterina, and Thorsten Joachims. [CAB: Continuous Adaptive Blending Estimator for Policy Evaluation and Learning](https://proceedings.mlr.press/v97/su19a). In *Proceedings of the 36th International Conference on Machine Learning*, 6005-6014, 2019.

14. Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudík. [Doubly Robust Off-policy Evaluation with Shrinkage](https://proceedings.mlr.press/v119/su20a.html). In *Proceedings of the 37th International Conference on Machine Learning*, 9167-9176, 2020.

15. Nathan Kallus and Angela Zhou. [Policy Evaluation and Optimization with Continuous Treatments](https://arxiv.org/abs/1802.06037). In *International Conference on Artificial Intelligence and Statistics*, 1243–1251. PMLR, 2018.

16. Aman Agarwal, Soumya Basu, Tobias Schnabel, and Thorsten Joachims. [Effective Evaluation using Logged Bandit Feedback from Multiple Loggers](https://arxiv.org/abs/1703.06180). In *Proceedings of the 23rd ACM SIGKDD international conference on Knowledge discovery and data mining*, 687–696, 2017.

17. Nathan Kallus, Yuta Saito, and Masatoshi Uehara. [Optimal Off-Policy Evaluation from Multiple Logging Policies](http://proceedings.mlr.press/v139/kallus21a.html). In *Proceedings of the 38th International Conference on Machine Learning*, 5247-5256, 2021.

18. Shuai Li, Yasin Abbasi-Yadkori, Branislav Kveton, S Muthukrishnan, Vishwa Vinay, and Zheng Wen. [Offline Evaluation of Ranking Policies with Click Models](https://arxiv.org/pdf/1804.10488). In *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery&Data Mining*, 1685–1694, 2018.

19. James McInerney, Brian Brost, Praveen Chandar, Rishabh Mehrotra, and Benjamin Carterette. [Counterfactual Evaluation of Slate Recommendations with Sequential Reward Interactions](https://arxiv.org/abs/2007.12986). In *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery&Data Mining*, 1779–1788, 2020.

20.  Yusuke Narita, Shota Yasui, and Kohei Yata. [Debiased Off-Policy Evaluation for Recommendation Systems](https://arxiv.org/abs/2002.08536). *arXiv preprint arXiv:2002.08536*, 2020.

21. Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, and Jure Leskovec. [Open Graph Benchmark: Datasets for Machine Learning on Graphs](https://arxiv.org/abs/2005.00687). In *Advances in Neural Information Processing Systems*. 2020.

</details>

<details>
<summary><strong>オープンソースプロジェクト</strong></summary>
本プロジェクトは **Open Graph Benchmark** ([[github](https://github.com/snap-stanford/ogb)] [[project page](https://ogb.stanford.edu)] [[paper](https://arxiv.org/abs/2005.00687)]) を参考にしています.
</details>
