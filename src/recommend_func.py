import pandas as pd
import numpy as np
from typing import List

import yaml

from src.text_preprocess import (Compose, ZenToHan, SpaceRemove, NumberRemove)
from src.embedding.encoder import Encoder
from src.embedding.bert_encoder import BertEncoder
from src.embedding.swem_encoder import SwemEncoder
from src.message_scraper import MessageScraper
from src import calculate_similarity
from typing import List
from src.metrics import timer


class Recommender(object):
    """
    質問文とその質問文が投稿されたタイムシフトから過去の質問を取得し，類似質問を計算するモジュール
    """

    def __init__(self, encoders: List[Encoder], message_scraper: MessageScraper):
        """
        :param encoders: 使用するSentence Embeddingsのエンコーダーのリスト
        :param message_scraper: 使用するMessageScraper
        """
        self.encoders = encoders
        self.message_scraper = message_scraper

    def get_recommend(self, text: str, channel_id: str, start_ts=None, days=7) -> pd.DataFrame:
        """
        質問文とその質問文のtsから過去の質問を取得し，類似度を計算する
        :param text: 質問文
        :param channel_id: 質問文が投稿されたチャンネルID
        :param start_ts: 質問文のts
        :param days: 過去何日間の質問データを取得するか
        :return:
        """
        df = self.message_scraper.get_messages(channel_id, start_ts=start_ts, days=days)
        df["answer"] = df.thread_ts.apply(
            lambda x: self.message_scraper.get_replies(channel_id=channel_id, message_ts=x))
        cos_sim_list = []
        for encoder in self.encoders:
            matrix = encoder.get_matrix(df["text"])
            vector = encoder.get_vector(text)
            cos_sim_list.append(calculate_similarity.most_similarity(matrix, vector))
        df["cos_sim"] = np.mean(cos_sim_list, axis=0)
        return df


def make():
    swem_transforms = Compose([ZenToHan(), NumberRemove(), SpaceRemove()])
    bert_transforms = Compose([ZenToHan(), SpaceRemove()])
    with open("./config.yml") as f:
        cfg = yaml.load(f)
    swem_encoder = SwemEncoder(transforms=swem_transforms, model_path="./data/model/jawiki.word_vectors.300d.bin")
    bert_encoder = BertEncoder(transforms=bert_transforms,
                               model_path="./data/model/training_bert_japanese/0_BERTJapanese/")
    sbert_encoder = BertEncoder(transforms=bert_transforms,
                                model_path="cl-tohoku/bert-base-japanese-whole-word-masking")
    message_scraper = MessageScraper(cfg)
    recommender = Recommender([swem_encoder, bert_encoder, sbert_encoder], message_scraper)
    return recommender
