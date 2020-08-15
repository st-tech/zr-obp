<p align="center">
  <img width="60%" src="./images/logo.png" />
</p>

**[ドキュメント](https://zr-obp.readthedocs.io/en/latest/)** | **[概要](#概要)** | **[インストール](#インストール)** | **[使用方法](#使用方法)** | **[参考](#参考)**  | **[Quickstart](https://github.com/st-tech/zr-obp/blob/master/examples/quickstart/quickstart.ipynb)** | **[Open Bandit Dataset](https://github.com/st-tech/zr-obp/blob/master/obd/README_JN.md)**

<details>
<summary><strong>Table of Contents</strong></summary>

- [概要](#概要)
  - [Open Bandit Dataset](#open-bandit-dataset)
  - [Open Bandit Pipeline](#open-bandit-pipeline)
    - [実装されているバンディットアルゴリズムとオフ方策推定量](#実装されているバンディットアルゴリズムとオフ方策推定量)
  - [トピックとタスク](#トピックとタスク)
- [インストール](#インストール)
  - [依存パッケージ](#依存パッケージ)
- [使用方法](#使用方法)
  - [(1) データの読み込みと前処理](#1-データの読み込みと前処理)
  - [(2) オフライン方策シミュレーション](#2-オフライン方策シミュレーション)
  - [(3) オフライン方策評価 （Off-Policy Evaluation）](#3-オフライン方策評価-off-policy-evaluation)
- [引用](#引用)
- [ライセンス](#ライセンス)
- [プロジェクトチーム](#プロジェクトチーム)
- [参考](#参考)
  - [論文](#論文)
  - [実装](#実装)

</details>

# 概要

## Open Bandit Dataset

*Open Bandit Dataset*は, バンディットアルゴリズムにまつわる研究を促進するための大規模公開実データです.
本データセットは, 日本最大のファッションEコマース企業である[株式会社ZOZO](https://corp.zozo.com/about/profile/)が提供しています.
同社が運営する大規模ファッションECサイト[ZOZOTOWN](https://zozo.jp/)では, 多腕バンディットアルゴリズムを用いてユーザーにファッションアイテムを推薦しています.
バンディットアルゴリズムによるファッションアイテム推薦の例は以下の通りです.

<p align="center">
  <img width="45%" src="./images/recommended_fashion_items.png" />
  <figcaption>
    <p align="center">
      図1. ZOZOTOWNにおけるファッションアイテムの推薦の例
    </p>
  </figcaption>
</p>

2019年11月下旬の7日間にわたる実験を行い, 全アイテム・男性用アイテム・女性用アイテムに対応する3つの「キャンペーン」でデータを収集しました.
それぞれのキャンペーンでは, 各ユーザーのインプレッションに対してランダム方策(Random)またはトンプソン抽出方策(Bernoulli Thompson Sampling)のいずれかを確率的にランダムに選択して適用しています.
図1はOpen Bandit Datasetの記述統計を示しています.

<p align="center">
  <img width="70%" src="./images/statistics_of_obd.png" />
  <figcaption>
    <p align="center">
      図2. Open Bandit Datasetのキャンペーンとデータ収集方策ごとの記述統計
    </p>
  </figcaption>
</p>

## Open Bandit Pipeline

*Open Bandit Pipeline*は, データセットの前処理・オフライン方策シミュレーション・オフ方策推定量の評価を簡単に行うためのパイプライン実装です. このパイプラインにより, 研究者はオフ方策推定量 (Off-Policy Estimator) の部分の実装に集中して現実的で再現性のある方法で他の手法との性能比較を行うことができるようになります.

<p align="center">
  <img width="90%" src="./images/overview.png" />
  <figcaption>
    <p align="center">
      図3. Open Bandit Pipelineの構成
    </p>
  </figcaption>
</p>

Open Bandit Pipeline は, 以下の主要モジュールで構成されています.

- **datasetモジュール**。このモジュールは, Open Bandit Dataset用のデータ読み込みクラスとデータの前処理するための柔軟なインターフェースを提供します. また人工データを生成するクラスも実装しています.
- **policyモジュール**: このモジュールは, バンディットアルゴリズムのためのインターフェイスを提供します. 加えて, いくつかの標準なバンディットアルゴリズムを実装しています.
- **simulatorモジュール**: このモジュールは, オフラインのバンディットシミュレーションを行うための関数を提供します.
- **opeモジュール**:　このモジュールは, いくつかの標準的なオフ方策推定量を実装しています. また新たにオフ方策推定量を実装するためのインターフェースを提供します.


### 実装されているバンディットアルゴリズムとオフ方策推定量

- バンディットアルゴリズム (**policy module**)
  - Online
    - Context-free
      - Random
      - Epsilon Greedy
      - Bernoulli Thompson Sampling
    - Contextual
      - Linear
        - Linear Epsilon Greedy
        - Linear Thompson Sampling [[Agrawal and Goyal. 2013]](http://proceedings.mlr.press/v28/agrawal13)
        - Linear Upper Confidence Bound [[Li et al. 2010]](https://dl.acm.org/doi/pdf/10.1145/1772690.1772758)
      - Logistic
        - Logistic Epsilon Greedy
        - Logistic Thompson Sampling [[Chapelle and Li. 2011]](https://papers.nips.cc/paper/4321-an-empirical-evaluation-of-thompson-sampling)
        - Logistic Upper Confidence Bound [[Mahajan et al. 2012]](https://dl.acm.org/doi/10.1145/2396761.2396767)
  - Offline (Off-Policy Learning) [[Dudík et al. 2014]](https://arxiv.org/abs/1503.02834)
    - Inverse Probability Weighting

- オフ方策推定量 (**ope module**)
  - Replay Method [[Li et al. 2011]](https://arxiv.org/abs/1003.5956)
  - Direct Method [[Beygelzimer and Langford 2009]](https://arxiv.org/abs/0812.4044)
  - Inverse Probability Weighting [[Precup et al. 2000]](https://scholarworks.umass.edu/cgi/viewcontent.cgi?article=1079&context=cs_faculty_pubs) [[Strehl et al. 2010]](https://arxiv.org/abs/1003.0120)
  - Self-Normalized Inverse Probability Weighting [[Swaminathan and Joachims. 2015]](https://papers.nips.cc/paper/5748-the-self-normalized-estimator-for-counterfactual-learning)
  - Doubly Robust [[Dudík et al. 2014]](https://arxiv.org/abs/1503.02834)
  - Switch Estimator [[Wang et al. 2016]](https://arxiv.org/abs/1612.01205)
  - More Robust Doubly Robust [[Farajtabar et al. 2018]](https://arxiv.org/abs/1802.03493)

私たちのパイプラインは, 上記のアルゴリズムや推定量に加えて柔軟なインターフェースも提供しています.
したがって研究者は, 独自のバンディットアルゴリズムやオフ方策推定量を容易に実装し, 我々のデータとパイプラインを用いてそれらの性能を評価することができます.
さらにパイプラインは, ログに記録されたバンディットフィードバックデータセットのためのインタフェースを含んでいます.
したがって, エンジニアやデータサイエンティストなどの実践者は, 独自のデータセットをパイプラインと組み合わせることで, 自社の設定・環境でバンディットアルゴリズムの性能を簡単に評価することができます.

## トピックとタスク

Open Bandit Dataset・Pipelineでは, 以下の研究テーマに関する実験評価を行うことができます.

- **バンディットアルゴリズムの評価 (Evaluation of Bandit Algorithms)**：我々の公開データには, ランダム方策によって収集された大規模なログデータが含まれています. このため, 大規模な実世界環境で新しいオンラインバンディットアルゴリズムの性能を評価することが可能です.


- **オフライン方策評価の評価 (Evaluation of Off-Policy Evaluation)**：我々の公開データは, 複数の方策を実システム上で走らせることによって生成されたログデータで構成されています. またそれらの方策の真の性能が含まれています. そのため, オフ方策推定量の推定精度の評価を行うことができます.


# インストール

以下の通り, `pip`を用いてOpen Bandit Pipelineをダウンロードすることが可能です.

```
pip install obp
```

また, 本リポジトリを直接cloneしてセットアップすることもできます.
```bash
git clone https://github.com/st-tech/zr-obp
cd zr-obp
python setup.py install
```


## 依存パッケージ
- **python>=3.7.0**
- matplotlib>=3.2.2
- numpy>=1.18.1
- pandas>=0.25.1
- pyyaml>=5.1
- seaborn>=0.10.1
- scikit-learn>=0.23.1
- scipy>=1.4.1
- tqdm>=4.41.1


# 使用方法

ここでは, Open Bandit Pipelineの使用法を説明します.
具体例として, リプレイ推定量とランダム方策によって生成されたログデータを用いて, トンプソン抽出方策の性能をオフラインで評価する例を使います.
以下に示すように, 約10行のコードでオフライン方策評価を行うことができます.

```python
# リプレイ推定量とランダム方策によって生成されたログデータを用いて, BernoulliTSの性能をオフラインで評価する
from obp.dataset import OpenBanditDataset
from obp.policy import BernoulliTS
from obp.simulator import run_bandit_simulation
from obp.ope import OffPolicyEvaluation, ReplayMethod

# (1) データの読み込みと前処理
dataset = OpenBanditDataset(behavior_policy='random', campaign='women')
bandit_feedback = dataset.obtain_batch_bandit_feedback()

# (2) オフライン方策シミュレーション
counterfactual_policy = BernoulliTS(n_actions=dataset.n_actions, len_list=dataset.len_list)
selected_actions = run_bandit_simulation(train=train, policy=counterfactual_policy)

# (3) オフライン方策評価
ope = OffPolicyEvaluation(train=train, ope_estimators=[ReplayMethod()])
estimated_policy_value = ope.estimate_policy_values(selected_actions=selected_actions)

# ランダム方策に対するトンプソン抽出方策の性能の改善率（相対クリック率）
relative_policy_value_of_bernoulli_ts = estimated_policy_value['rm'] / test['reward'].mean()
print(relative_policy_value_of_bernoulli_ts) # 1.120574...
```

同じ例を使った詳細な実装例は[quickstart](https://github.com/st-tech/zr-obp/blob/master/examples/quickstart/)にあり, 実際に動かして試してみることが可能です.
以下, 重要な要素について詳細に説明します.

## (1) データの読み込みと前処理

Open Bandit Dataset用のデータ読み込みインターフェースを用意しています.
これにより, Open Bandit Datasetの読み込みや前処理を簡潔に行うことができます.

```python
# 「女性向けキャンペーン」においてランダム方策が集めたログデータを読み込む.
# OpenBanditDatasetクラスにはデータを収集した方策とキャンペーンを指定する.
dataset = OpenBanditDataset(behavior_policy='random', campaign='women')
# オフライン方策シミュレーション用データを得る.
bandit_feedback = dataset.obtain_batch_bandit_feedback()

print(bandit_feedback.keys())
# dict_keys(['n_rounds', 'n_actions', 'action', 'position', 'reward', 'pscore', 'context', 'action_context'])
```

`obp.dataset.OpenBanditDataset` クラスの `pre_process` メソッドに, 独自の特徴量エンジニアリングを実装することもできます.
[`./examples/examples_with_obd/custom_dataset.py`](https://github.com/st-tech/zr-obp/blob/master/examples/examples_with_obd/custom_dataset.py)には, 新しい特徴量エンジニアリングを実装する例を示しています.
また, `obp.dataset.BaseBanditDataset`クラスのインターフェースに従って新たなクラスを実装することで, 将来公開されるであろうOpen Bandit Dataset以外のバンディットデータセットを扱うこともできます.

## (2) オフライン方策シミュレーション

前処理の後は, 次のようにして**オフライン方策シミュレーション**を実行します.

```python
# オフライン方策シミュレーション.
# 評価対象の反実仮想アルゴリズム. ここでは, トンプソン抽出方策の性能をオフライン評価する.
# 研究者が独自に実装したバンディット方策を用いることもできる.
counterfactual_policy = BernoulliTS(n_actions=dataset.n_actions, len_list=dataset.len_list)
# シミュレーション用データ(bandit_feedback)上でトンプソン抽出方策を動作させる.
selected_actions = run_bandit_simulation(bandit_feedback=bandit_feedback, policy=counterfactual_policy)
```

オフライン方策シミュレーションを行うための関数である `obp.simulator.run_bandit_simulation`は `obp.policy.BanditPolicy` クラスと `bandit_feedback` (シミュレーション用データを格納したdictionary) を入力として受け取り, 与えられたバンディット方策（ここでは`BernoulliTS`）をシミュレーション用データ上で動作させます. そしてシミュレーション中にバンディット方策が選択したアクション (ここでは, `selected_actions`) を返します. またユーザーは[`./obp/policy/base.py`](https://github.com/st-tech/zr-obp/blob/master/obp/policy/base.py)に実装されているインターフェースに従うことで独自のバンディットアルゴリズムを実装し, その性能を評価することができます.


## (3) オフライン方策評価 （Off-Policy Evaluation）

最後のステップは, オフラインでのバンディットシミュレーションによって生成されたログデータを用いてバンディットアルゴリズムの性能を推定する**オフライン方策評価**です.
我々のパイプラインでは, 以下のような手順でオフライン方策評価を行うことができます.

```python
# オフライン方策シミュレーションの結果に基づき, リプレイ推定量を用いてトンプソン抽出方策の性能をオフライン評価する.
# OffPolicyEvaluationクラスには, シミュレーションに用いたデータセットと用いる推定量を渡す（複数設定可）.
ope = OffPolicyEvaluation(bandit_feedback=bandit_feedback, ope_estimators=[ReplayMethod()])
estimated_policy_value = ope.estimate_policy_values(selected_actions=selected_actions)
print(estimated_policy_value) # {'rm': 0.005155..}　オフ方策推定量ごとの推定値を含んだ辞書.

# トンプソン抽出方策の性能の推定値とランダム方策の真の性能を比較する.
relative_policy_value_of_bernoulli_ts = estimated_policy_value['rm'] / bandit_feedback['reward'].mean()
# オフライン方策評価によって, トンプソン抽出方策の性能はランダム方策の性能を12.05%上回ると推定された.
print(relative_policy_value_of_bernoulli_ts) # 1.120574...
```
ユーザーは独自のオフ方策推定量を `obp.ope.BaseOffPolicyEstimator` クラスのインターフェースに従って実装することができます.
これにより新たなオフ方策推定量の推定精度をすぐに検証することが可能です.
また, `obp.ope.OffPolicyEvaluation`の`ope_estimators`引数に複数のオフ方策推定量を設定することによって, 複数の推定量による推定値を同時に得ることも可能です. `bandit_feedback['reward'].mean()` は観測された報酬の経験平均値（オン方策推定）であり, ランダム方策の真の性能を表します.


# 引用
本リポジトリを活用して論文を執筆された場合, 以下の論文を引用していただくようよろしくお願いいたします.

```
# TODO: add bibtex
@article{
}
```


# ライセンス
このプロジェクトはApache 2.0ライセンスを採用しています.  詳細は, [LICENSE](https://github.com/st-tech/zr-obp/blob/master/LICENSE) を参照してください.


# プロジェクトチーム

- [齋藤優太](https://usaito.github.io/) (**Main Contributor**; 半熟仮想株式会社 / 東京工業大学)
- [粟飯原俊介](https://www.linkedin.com/in/shunsukeaihara/) (半熟仮想株式会社 / ZOZO Technologies, Inc.)
- 松谷恵 (ZOZO Technologies, Inc.)
- [成田悠輔](https://www.yusuke-narita.com/) (半熟仮想株式会社 / イェール大学)


# 参考

## 論文
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

13.  Yusuke Narita, Shota Yasui, and Kohei Yata. [Off-policy Bandit and Reinforcement Learning](https://arxiv.org/abs/2002.08536). *arXiv preprint arXiv:2002.08536*, 2020.

14. Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, and Jure Leskovec. [Open Graph Benchmark: Datasets for Machine Learning on Graphs](https://arxiv.org/abs/2005.00687). *arXiv preprint arXiv:2005.00687*, 2020.

## 実装
本プロジェクトは **Open Graph Benchmark** ([[github](https://github.com/snap-stanford/ogb)] [[project page](https://ogb.stanford.edu)] [[paper](https://arxiv.org/abs/2005.00687)]) を参考にしています.
