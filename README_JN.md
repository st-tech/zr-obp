# Open Bandit Dataset & Pipeline

**[ドキュメント](https://zr-obp.readthedocs.io/en/latest/)** | **[概要](#概要)** | **[インストール](#インストール)** | **[使用方法](#使用方法)** | **[参考](#参考)**  | **[Quickstart](./examples/quickstart/quickstart.ipynb)** | **[Open Bandit Dataset](./obd/README_JN.md)**

## 概要

### **Open Bandit Dataset**

*Open Bandit Dataset*は, バンディットアルゴリズムにまつわる研究を促進するための大規模公開実データです.
本データセットは, 日本最大のファッションEコマース企業である[株式会社ZOZO](https://corp.zozo.com/about/profile/)が提供しています.
同社が運営する大規模ファッションECサイト[ZOZOTOWN](https://zozo.jp/)では, 多腕バンディットアルゴリズムを用いてユーザーにファッションアイテムを推薦しています.
バンディットアルゴリズムによるファッションアイテム推薦の例は以下の通りです.

<p align="center">
  <img width="40%" src="./images/recommended_fashion_items.png" />
  <figcaption>
    <p align="center">
      図1. ZOZOTOWNにおけるファッションアイテムの推薦の例
    </p>
  </figcaption>
</p>

2019年11月下旬の7日間にわたる実験を行い, 全アイテム・男性用アイテム・女性用アイテムに対応する3つの「キャンペーン」でデータを収集しました.
それぞれのキャンペーンでは, 各ユーザーのインプレッションに対してランダム方策 (Random)またはトンプソン抽出方策 (Bernulli Thompson Sampling)のいずれかを確率的にランダムに選択して適用しています.
図1はOpen Bandit Datasetの記述統計を示しています.

<p align="center">
  <img width="70%" src="./images/statistics_of_obd.png" />
  <figcaption>
    <p align="center">
      図2. Open Bandit Datasetのキャンペーンとデータ収集方策ごとの記述統計
    </p>
  </figcaption>
</p>


### **Open Bandit Pipeline**

*Open Bandit Pipeline*は, データセットの前処理・オフライン方策シミュレーション・オフ方策推定量の評価を簡単に行うためのパイプライン実装です. このパイプラインにより, 研究者はオフ方策推定量 (Off-Policy Estimator) の部分の実装に集中して現実的で再現性のある方法で他の手法との性能比較を行うことができるようになります.

<p align="center">
  <img width="85%" src="./images/overview.png" />
  <figcaption>
    <p align="center">
      図3. Open Bandit Pipelineの概要. (i) Open Bandit Dataset (ii) データセットの読み込みと前処理 (iii) オフライン方策シミュレーター (iv) オフ方策推定量（研究者自身が実装すべき部分） (v) 推定量の性能の評価
    </p>
  </figcaption>
</p>


### トピックとタスク

Open Bandit Dataset・Pipelineでは, 以下の研究テーマに関する実験評価を行うことができます.

- **バンディットアルゴリズムの評価 (Evaluation of Bandit Algorithms)**：我々の公開データには, ランダム方策によって収集された大規模なログデータが含まれています. このため, 大規模な実世界環境で新しいオンラインバンディットアルゴリズムの性能を評価することが可能です.


- **オフライン方策評価の評価 (Evaluation of Off-Plicy Evaluation)**：我々の公開データは, 複数の方策を実システム上で走らせることによって生成されたログデータで構成されています. またそれらの方策の真の性能が含まれています. そのため, オフ方策推定量の評価を行うことができます.


## インストール

以下の通り, `pip`を用いてOpen Bandit Pipelineをダウンロードすることが可能です.

```
pip install obp
```

また, 本リポジトリを直接cloneしてセットアップすることもできます.
```bash
git clone https://github.com/st-tech/zr-obp
cd obp
python setup.py install
```


### 依存パッケージ
- **python>=3.7.0**
- numpy>=1.18.1
- pandas>=0.25.1
- scipy>=1.4.1
- scikit-learn>=0.23.1
- tqdm>=4.41.1
- pyyaml>=5.1


## 使用方法

ここでは, Open Bandit Pipelineの使用法を説明します.
具体例として, リプレイ推定量とランダム方策によって生成されたログデータを用いて, トンプソン抽出方策の性能をオフラインで評価する例を使います.
以下に示すように, 約10行のコードでオフライン方策評価を行うことができます.

```python
# リプレイ推定量とランダム方策によって生成されたログデータを用いて, BernoulliTSの性能をオフラインで評価する
from obp.dataset import OpenBanditDataset
from obp.policy import BernoulliTS
from obp.simulator import run_bandit_simulation
from obp.ope import OffPolicyEvaluation ReplayMethod

# (1) データの読み込みと前処理
dataset = OpenBanditDataset(behavior_policy='random', campaign='all')
train, test = dataset.split_data(test_size=0.3, random_state=42)

# (2) オフライン方策シミュレーション
counterfactual_policy = BernoulliTS(n_actions=dataset.n_actions, len_list=dataset.len_list)
selected_actions = run_bandit_simulation(train=train, policy=counterfactual_policy)

# (3) オフライン方策評価
ope = OffPolicyEvaluation(train=train, ope_estimators=[ReplayMethod()])
estimated_policy_value = ope.estimate_policy_values(selected_actions=selected_actions)

# ランダム方策に対するトンプソン抽出方策の性能の改善率（相対クリック率）
relative_policy_value_of_bernoulli_ts = estimated_policy_value['rm'] / test['reward'].mean()
print(relative_policy_value_of_bernoulli_ts) # 1.21428...
```

同じ例を使った詳細な実装例は[quickstart](./examples/quickstart/quickstart.ipynb)にあり, 実際に動かして試してみることが可能です.
以下, 重要な要素について詳細に説明します.

### (1) データの読み込みと前処理

Open Bandit Dataset用のデータ読み込みインターフェースを用意しています.
これにより, Open Bandit Datasetの前処理や標準化されたデータの分割を簡潔に行うことができます.

```python
# 「女性向けキャンペーン」においてランダム方策が集めたログデータを読み込む.
# OpenBanditDatasetクラスにはデータを収集した方策とキャンペーンを指定する.
dataset = OpenBanditDataset(behavior_policy='random', campaign='all')
# データセットを70%のオフライン方策シミュレーション用データと30%のオフライン方策評価用データに分割する.
train, test = dataset.split_data(test_size=0.3, random_state=0)

print(train.keys())
# dict_keys(['n_rounds', 'n_actions', 'action', 'position', 'reward', 'pscore', 'context', 'action_context'])
```

`OpenBanditDataset` クラスの `pre_process` メソッドに, 独自の特徴量エンジニアリングを実装することもできます.
[`./examples/obd/dataset.py`](./examples/obd/dataset.py)には, 新しい特徴量エンジニアリングを実装する例を示しています.

また, `./obp/dataaset.py`の`BaseBanditDataset`クラスのインターフェースに従って新たなクラスを実装することで, 将来公開されるであろうOpen Bandit Dataset以外のバンディットデータセットを扱うこともできます.

### (2) オフライン方策シミュレーション

前処理の後は, 次のようにして**オフライン方策シミュレーション**を実行します.

```python
# オフライン方策シミュレーション.
# 評価対象の反実仮想アルゴリズム. ここでは, トンプソン抽出方策の性能をオフライン評価する.
# 研究者が独自に実装したバンディット方策を用いることもできる.
counterfactual_policy = BernoulliTS(n_actions=dataset.n_actions, len_list=dataset.len_list)
# シミュレーション用データ(train)上でトンプソン抽出方策を動作させる.
selected_actions = run_bandit_simulation(train=train, policy=counterfactual_policy)
```

オフライン方策シミュレーションを行うための関数である `run_bandit_simulation`は `BanditPolicy` クラスと `train` (シミュレーション用データを格納したdictionary) を入力として受け取り, 与えられたバンディット方策（ここでは`BernoulliTS`）をシミュレーション用データ上で動作させます. そしてシミュレーション中に設定したバンディット方策が選択したアクション (ここでは, `selected_actions`) を返します.

ユーザーは[`./obp/policy/contextfree.py`](./obp/policy/contextfree.py)の`BaseContextFreePolicy`や[`./obp/policy/contextual.py`](./obp/policy/contextual.py)の`BaseContexttualPolicy`のインターフェースに従うことで, 独自のバンディットアルゴリズムを実装することができます.


### (3) オフライン方策評価 （Off-Policy Evaluation）

最後のステップは, オフラインでのバンディットシミュレーションによって生成されたログデータを用いてバンディットアルゴリズムの性能を推定する**オフライン方策評価**です.
我々のパイプラインでは, 以下のような手順でオフライン方策評価を行うことができます.

```python
# オフライン方策シミュレーションの結果に基づき, リプレイ推定量を用いてトンプソン抽出方策の性能をオフライン評価する.
# OffPolicyEvaluationクラスには, シミュレーションに用いたデータセットと用いる推定量を渡す（複数設定可）.
ope = OffPolicyEvaluation(train=train, ope_estimators=[ReplayMethod()])
estimated_policy_value = ope.estimate_policy_values(selected_actions=selected_actions)
print(estimated_policy_value) # {'rm': 0.003717..}

# トンプソン抽出方策の性能の推定値とランダム方策の真の性能を比較する.
relative_policy_value_of_bernoulli_ts = estimated_policy_value['rm'] / test['reward'].mean()
# オフライン方策評価によって, トンプソン抽出方策の性能はランダム方策の性能を21.4%上回ると推定された.
print(relative_policy_value_of_bernoulli_ts) # 1.21428...
```
ユーザーは独自のオフ方策推定量を `BaseOffPolicyEstimator` クラスのインターフェースに従って実装することができます.
また, `OffPolicyEvaluation`の`ope_estimators`引数に複数のオフ方策推定量を設定することによって, 複数の推定量による推定値を同時に得ることも可能です. `test['reward'].mean()` は観測された報酬の経験平均値であり, ランダム方策の真の性能を表します.


## 引用
本リポジトリを活用して論文を執筆された場合, 以下の論文を引用していただくようお願いいたします.

```
# TODO: add bibtex
@article{
}
```


## ライセンス
このプロジェクトはApache 2.0ライセンスを採用しています.  詳細は, [LICENSE](LICENSE) を参照してください.


## 主要コントリビューター

- [Yuta Saito](https://usaito.github.io/)


## 参考

### 論文
1. Alina Beygelzimer and John Langford. The offset tree for learning with partial labels. In
Proceedings of the 15th ACM SIGKDD international conference on Knowledge discovery and data mining, pages 129–138, 2009.

2. Olivier Chapelle and Lihong Li. An empirical evaluation of thompson sampling. In Advances in neural information processing systems, pages 2249–2257, 2011.

3. Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li. Doubly Robust Policy Evaluation and Optimization. Statistical Science, 29:485–511, 2014.

4. Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, and Jure Leskovec. Open graph benchmark: Datasets for machine learning on graphs. arXiv preprint arXiv:2005.00687, 2020.

5. Lihong Li, Wei Chu, John Langford, and Xuanhui Wang. Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms. In Proceedings of the Fourth ACM International Conference on Web Search and Data Mining, pages 297–306, 2011.

6. Yusuke Narita, Shota Yasui, and Kohei Yata. Off-policy Bandit and Reinforcement Learning. arXiv preprint arXiv:2002.08536, 2020.

7. Alex Strehl, John Langford, Lihong Li, and Sham M Kakade. Learning from Logged Implicit Exploration Data. In Advances in Neural Information Processing Systems, pages 2217–2225, 2010.

8. Adith Swaminathan and Thorsten Joachims. The Self-normalized Estimator for Counterfactual Learning. In Advances in Neural Information Processing Systems, pages 3231–3239, 2015.

### 実装
本プロジェクトは **Open Graph Benchmark** ([[github](https://github.com/snap-stanford/ogb)] [[project page](https://ogb.stanford.edu)] [[paper](https://arxiv.org/abs/2005.00687)]) を大いに参考しています.
