import pandas as pd
import numpy as np
from contextlib import contextmanager
import time
from datetime import datetime, timedelta
import os
from glob import glob
from src.embedding.encoder import Encoder
from src import calculate_similarity
from typing import List
from src.embedding.swem_encoder import SwemEncoder
from src.embedding.bert_encoder import BertEncoder
from src.text_preprocess import (Compose, ZenToHan, SpaceRemove, NumberRemove)


@contextmanager
def timer(name: str) -> None:
    """
    時間を測定するコンテキストマネージャー
    :param name:
    :return:
    """
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.3f} s')


def make_evaluation_data(index=100, days=7, channel="G01ADNFSRS4") -> None:
    """
    評価データを作成する関数
    question.txtは対象の質問文, data.csvは過去の質問データ
    :param index: {channel}.csvのindexの質問を対象の質問文とする
    :param days: 対象の質問文が投稿されてから過去days日分を過去の質問データとする
    :param channel: 対象のチャンネル
    :return:
    """
    all_data = pd.read_csv(f"./data/{channel}.csv")

    all_data["time"] = all_data.thread_ts.apply(datetime.fromtimestamp)
    raw = all_data.iloc[index]
    data = all_data[(all_data.time > raw.time - timedelta(days=days)) & (all_data.time < raw.time)].reset_index(
        drop=True)
    data["label"] = 0

    os.makedirs(f"./data/dataset/{channel}_{index}_days{days}", exist_ok=True)
    with open(f"./data/dataset/{channel}_{index}_days{days}/question.txt", "w") as f:
        f.write(raw.text)
    with open(f"./data/dataset/{channel}_{index}_days{days}/thread_ts.txt", "w") as f:
        f.write(str(raw.thread_ts))
    data.to_csv(f"./data/dataset/{channel}_{index}_days{days}/data.csv", index=False)


def mean_reciprocal_rank(y_true, y_pred):
    """
    MRR(Mean Reciprocal Rank)を計算する
    :param y_true: 0か1からなるラベルのリスト, example) [1, 0, 0, 1]
    :param y_pred: 降順でランキングが割り当てられる, example) [0.9, 0.3, 0.2, 0.8]
    :return: [0, 1]の範囲の値
    """
    sort_idx = np.argsort(y_pred)
    y_true_sorted = y_true[sort_idx][::-1]
    ranking = np.where(y_true_sorted)[0].tolist()
    if ranking:
        return 1 / (ranking[0] + 1)
    return 0


def mean_average_precision(y_true, y_pred):
    """
    MAP(Mean Average Precision)を計算する
    :param y_true: 0か1からなるラベルのリスト, example) [1, 0, 0, 1]
    :param y_pred: 降順でランキングが割り当てられる, example) [0.9, 0.3, 0.2, 0.8]
    :return: [0, 1]の範囲の値
    """
    sort_idx = np.argsort(y_pred)
    y_true_sorted = y_true[sort_idx][::-1]
    ranking = np.where(y_true_sorted)[0]
    return np.mean([(i + 1) / (rank + 1) for i, rank in enumerate(ranking)])


def precision_at_n(y_true, y_pred, n=5):
    """
    Pre@N(Precision@N)を計算する
    :param y_true: 0か1からなるラベルのリスト, example) [1, 0, 0, 1]
    :param y_pred: 降順でランキングが割り当てられる, example) [0.9, 0.3, 0.2, 0.8]
    :return: [0, 1]の範囲の値
    """
    n = min(n, len(y_true))
    sort_idx = np.argsort(y_pred)
    y_true_sorted = y_true[sort_idx][::-1][:n]
    return np.sum(y_true_sorted) / n


def metric(encoder: Encoder):
    """
    ./data/dataset 下のデータの評価を行う
    :param encoder: 対象のEncoder(SWEM, BERT, SBERT)
    :return:
    """
    mrr = []
    ap = []
    precision_1 = []
    precision_5 = []
    precision_10 = []
    for dir in glob("./data/dataset/*"):
        with open(f"{dir}/question.txt", "r") as f:
            text = f.read()
        data = pd.read_csv(f"{dir}/data.csv")
        vector = encoder.get_vector(text)
        matrix = encoder.get_matrix(data["text"])
        data["cos_sim"] = calculate_similarity.most_similarity(matrix, vector)
        mrr.append(mean_reciprocal_rank(data["label"], data["cos_sim"]))
        ap.append(mean_average_precision(data["label"], data["cos_sim"]))
        precision_1.append(precision_at_n(data["label"], data["cos_sim"], n=1))
        precision_5.append(precision_at_n(data["label"], data["cos_sim"], n=5))
        precision_10.append(precision_at_n(data["label"], data["cos_sim"], n=10))

    print(f"MRR: {np.mean(mrr)}, "
          f"MAP: {np.mean(ap)}, "
          f"Precision@1: {np.mean(precision_1)}, "
          f"Precision@5: {np.mean(precision_5)}, "
          f"Precision@10: {np.mean(precision_10)}")


def metric_ensemble(encoder_list: List[Encoder]):
    """
    ./data/dataset 下のデータの評価を行う．各手法のコサイン類似度の平均でアンサンブルを行う
    :param encoder_list: 対象のEncoder(SWEM, BERT, SBERT)のリスト
    :return:
    """
    mrr = []
    ap = []
    precision_1 = []
    precision_5 = []
    precision_10 = []
    for dir in glob("./data/dataset/*"):
        with open(f"{dir}/question.txt", "r") as f:
            text = f.read()
        data = pd.read_csv(f"{dir}/data.csv")
        tmp = []
        for encoder in encoder_list:
            vector = encoder.get_vector(text)
            matrix = encoder.get_matrix(data["text"])
            tmp.append(calculate_similarity.most_similarity(matrix, vector))
        data["cos_sim"] = np.mean(tmp, axis=0)
        mrr.append(mean_reciprocal_rank(data["label"], data["cos_sim"]))
        ap.append(mean_average_precision(data["label"], data["cos_sim"]))
        precision_1.append(precision_at_n(data["label"], data["cos_sim"], n=1))
        precision_5.append(precision_at_n(data["label"], data["cos_sim"], n=5))
        precision_10.append(precision_at_n(data["label"], data["cos_sim"], n=10))

    print(f"MRR: {np.mean(mrr):.3f}, "
          f"MAP: {np.mean(ap):.3f}, "
          f"Precision@1: {np.mean(precision_1):.3f}, "
          f"Precision@5: {np.mean(precision_5):.3f}, "
          f"Precision@10: {np.mean(precision_10):.3f}")

    # print(mrr)
    # print(ap)
    # print(precision_1)
    # print(precision_5)
    # print(precision_10)


def eda_metric(path: str):
    """
    path下の評価データのコサイン類似度の結果を取得する
    :param path: 評価するデータセットのパス
    :return: 評価データの対象質問文, コサイン類似度の情報
    """
    transforms = Compose([ZenToHan(), NumberRemove(), SpaceRemove()])
    encoder1 = SwemEncoder(transforms=transforms, model_path="./data/model/jawiki.word_vectors.300d.bin")
    transforms = Compose([ZenToHan(), SpaceRemove()])
    encoder2 = BertEncoder(transforms=transforms, model_path="cl-tohoku/bert-base-japanese-whole-word-masking")
    encoder3 = BertEncoder(transforms=transforms, model_path="./data/model/training_bert_japanese/0_BERTJapanese/")
    encoder_list = {
        "SWEM": encoder1,
        "BERT": encoder2,
        "SBERT": encoder3
    }

    dir = "./data/dataset/" + path
    with open(f"{dir}/question.txt", "r") as f:
        text = f.read()
    data = pd.read_csv(f"{dir}/data.csv")
    tmp = []
    for key in encoder_list:
        vector = encoder_list[key].get_vector(text)
        matrix = encoder_list[key].get_matrix(data["text"])
        output = calculate_similarity.most_similarity(matrix, vector)
        tmp.append(output)
        data[key] = output
    data["Ensemble"] = np.mean(tmp, axis=0)
    return text, data


def eda(data: pd.DataFrame):
    """
    eda_metricで得たdataを可視化するメソッドver.1
    :param data:
    :return:
    """
    keys = ["SWEM", "BERT", "SBERT", "Ensemble"]

    print(f"Label Count: {data['label'].sum()}, Query Count: {len(data)}")

    def _output(key, n=5):
        label_list = data.sort_values(key, ascending=False)["label"].tolist()[:n]
        text_list = data.sort_values(key, ascending=False)["text"].tolist()[:n]
        cos_list = data.sort_values(key, ascending=False)[key].tolist()[:n]
        true_list = [i + 1 for i, label in enumerate(data.sort_values(key, ascending=False)["label"].tolist()) if label]
        print(f"True List: {true_list}")
        for label, text, cos in zip(label_list, text_list, cos_list):
            print(label, text.replace("\n", "[SEP]"), cos)

    for key in keys:
        print(f"---------{key}--------")
        _output(key)
        print(f"----------------------")


def eda2(data: pd.DataFrame):
    """
    eda_metricで得たdataを可視化するメソッドver.2
    :param data:
    :return:
    """
    print(f"Label Count: {data['label'].sum()}, Query Count: {len(data)}")
    keys = ["SWEM", "BERT", "SBERT", "Ensemble"]
    for key in keys:
        data[f"{key}_rank"] = data[key].rank(ascending=False)

    correct_data = data[data.label == 1]
    for text, swem, bert, sbert, ensemble in zip(correct_data["text"], correct_data["SWEM_rank"],
                                                 correct_data["BERT_rank"], correct_data["SBERT_rank"],
                                                 correct_data["Ensemble_rank"]):
        print(text.replace("\n", "[SEP]"), int(swem), int(bert), int(sbert), int(ensemble))


def get_predict_time():
    """
    各手法の実行速度の測定
    :return:
    """
    swem_transforms = Compose([ZenToHan(), NumberRemove(), SpaceRemove()])
    encoder1 = SwemEncoder(transforms=swem_transforms, model_path="./data/model/jawiki.word_vectors.300d.bin")
    transforms = Compose([ZenToHan(), SpaceRemove()])
    encoder2 = BertEncoder(transforms=transforms, model_path="cl-tohoku/bert-base-japanese-whole-word-masking")
    encoder3 = BertEncoder(transforms=transforms, model_path="./data/model/training_bert_japanese/0_BERTJapanese/")
    with timer("SWEM"):
        metric_ensemble([encoder1])
    with timer("BERT"):
        metric_ensemble([encoder2])
    with timer("SBERT"):
        metric_ensemble([encoder3])
