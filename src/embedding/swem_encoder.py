from gensim.models import Word2Vec, KeyedVectors
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Union, List
from src.embedding.encoder import Encoder
from src.text_preprocess import Tokenizer


class SwemEncoder(Encoder):
    """
    SWEMを計算するエンコーダー
    """

    def __init__(self, model_path: str, transforms=None, pooling_type="mean", n=2) -> None:
        """
        :param model_path: モデルのパス(*.bin)
        :param transforms: テキストの前処理(text_preprocess.pyから)
        :param pooling_type: ["mean", "max", "concat", "hier"]のいずれか
        :param n: SWEM-hierのときのウィンドウサイズ
        """
        pooling_list = ["mean", "max", "concat", "hier"]
        assert pooling_type in pooling_list, f"{pooling_type} is a non-existent pooling_type {pooling_list}"
        super().__init__(model_path, transforms=transforms, pooling_type=pooling_type)
        self.model = KeyedVectors.load(model_path)
        self.tokenizer = Tokenizer(hinshi_list=["動詞", "名詞", "形容詞"], split_mode="A")
        self.n = n

    def _get_word_embeddings(self, sentence: str) -> np.array:
        """
        単語の分散表現のリストを取得する
        :param sentence: 単語リスト
        :return: np.array, shape(len(word_list), self.vector_size)
        """
        word_list = self.tokenizer(sentence).split()
        np.random.seed(abs(hash(sentence)) % (10 ** 8))
        vectors = []

        for word in word_list:
            if word in self.model.vocab:
                vectors.append(self.model[word])
            else:
                vectors.append(np.random.uniform(-0.01, 0.01, self.model.vector_size))

        if not vectors:
            return np.random.uniform(-0.01, 0.01, self.model.vector_size)

        if self.pooling_type == "mean":
            return np.mean(vectors, axis=0)
        elif self.pooling_type == "max":
            return np.max(vectors, axis=0)
        elif self.pooling_type == "concat":
            return np.r_[np.mean(vectors, axis=0), np.max(vectors, axis=0)]
        elif self.pooling_type == "hier":
            n = self.n
            text_len = len(vectors)
            if n > text_len:
                n = text_len
            window_average_pooling_vec = [np.mean(vectors[i:i + n], axis=0) for i in range(text_len - n + 1)]
            return np.max(window_average_pooling_vec, axis=0)
        return np.array(vectors)

    def get_vector(self, sentence: str) -> np.array:
        return self._get_word_embeddings(sentence)

    def get_matrix(self, sentences: Union[List[str], pd.Series]) -> np.array:
        swems = []
        for sentence in sentences:
            swems.append(self._get_word_embeddings(sentence))

        return np.array(swems)
