# 類似質問応答システム

## Directory

```
.
├── main.py # 実験の実行
│
├── data
│   ├── dataset # 評価用のデータセットのフォルダ
│   │   └── *
│   ├── model
│   │   ├── training_bert_japanese # 学習済みSentence BERTモデル
│   │   │   └── *
│   │   └── jawiki.word_vectors.300d.bin # 学習済みSentence BERTモデル
│   ├── C01ARMDPQF7.csv
│   └── G01ADNFSRS4.csv
│
├── src
│   ├── embedding # 各手法のモジュール
│   │   └── *.py
│   ├── calculate_similarity.py # 類似度の計算
│   ├── message_scraper.py # 質問データを取得するモジュール
│   ├── metrics.py # 評価データの作成や実験，可視化など
│   ├── recommend_func.py # get_recommendで過去の質問データの取得 -> cos類似度の計算まで
│   └── text_preprocess.py # テキストの前処理関係のモジュール(albumentationsと同じような使い方)
│
├── config.yml # botのトークンの情報
├── docker-compose.yml
├── Dockerfile
├── LICENSE
├── README.md
└── requirements.txt
```

## 開発環境

### Python Version

```
Python 3.7.4
```

### Library

```
certifi
click
fastprogress
japanize-matplotlib
jupyter
logzero
matplotlib
mecab-python3
mojimoji
neologdn
numpy
pandas
scikit-learn
scipy
seaborn
tokenizers
torch
torchtext
torchvision
tqdm
transformers
emoji
gensim
sudachipy
SudachiDict-core
unidic-lite
fugashi
ipadic
ijson
```

## 各実験手順

### Dockerの起動
Docker起動
```shell
$ docker build -t {イメージ名} . 
$ docker run -itd -v $(pwd):/code --name {コンテナ名} {イメージ名}
```
コンテナにログイン
```shell
$ docker exec -it {コンテナ名} /bin/bash
```
実行
```shell
$ python3 main.py
```

### チャンネルのデータセットの作成(C01ARMDPQF7.csvやG01ADNFSRS4.csvの作り方)

```python
import yaml
from src.message_scraper import MessageScraper

with open("./config.yml") as f:
    cfg = yaml.load(f)
scraper = MessageScraper(cfg)
channel_id = "C01ARMDPQF7"  # channelのid
df = scraper.get_messages(channel_id=channel_id)
df["answer"] = df.thread_ts.apply(lambda x: scraper.get_replies(channel_id=channel_id, message_ts=x))
df.to_csv(f"./data/{channel_id}.csv", index=False)
```

### 評価用データの作成(./data/dataset/)

```python
from src import metrics

metrics.make_evaluation_data(index=100, days=7, channel="G01ADNFSRS4")
```

### データの評価

```python
from src.embedding.swem_encoder import SwemEncoder
from src.embedding.bert_encoder import BertEncoder
from src.text_preprocess import (Compose, ZenToHan, SpaceRemove, NumberRemove)
from src import metrics

transforms = Compose([ZenToHan(), NumberRemove(), SpaceRemove()])
encoder1 = SwemEncoder(transforms=transforms, model_path="./data/model/jawiki.word_vectors.300d.bin")
transforms = Compose([ZenToHan(), SpaceRemove()])
encoder2 = BertEncoder(transforms=transforms, model_path="cl-tohoku/bert-base-japanese-whole-word-masking")
transforms = Compose([ZenToHan(), SpaceRemove()])
encoder3 = BertEncoder(transforms=transforms, model_path="./data/model/training_bert_japanese/0_BERTJapanese/")
metrics.metric_ensemble([encoder1, encoder2, encoder3])  # 複数のエンコーダーを使用するとアンサンブルになる
```

### 各Sentence Embeddingsの推論速度の評価

```python
from src import metrics

metrics.get_predict_time()
```

### 評価データの可視化

パターン1

```python
from src import metrics

question, data = metrics.eda_metric("G01ADNFSRS4_50_days7")  # ./data/dataset下のフォルダ名
print(question)
metrics.eda1(data)
```

パターン2

```python
from src import metrics

question, data = metrics.eda_metric("G01ADNFSRS4_50_days7")  # ./data/dataset下のフォルダ名
print(question)
metrics.eda2(data)
```

### 質問文とその質問文のtsから過去の質問データとの類似度を計算する
pandas形式で結果が返される
```python
from src import recommend_func
recommender = recommend_func.make()
df = recommender.get_recommend("UML図について質問です。UMLにおいてfinal修飾子の表記方法がわかりません。教えていただけないでしょうか。", "C01ARMDPQF7", start_ts=1601014104.0468)
```

## 参照推奨文献
- [Huggingface Transformrs](https://huggingface.co/transformers/)
- [pandas](https://pandas.pydata.org/docs/)
- [Slack API](https://api.slack.com/)