from abc import ABCMeta, abstractmethod
from typing import Union, List

import pandas as pd
import numpy as np


class Encoder(metaclass=ABCMeta):

    def __init__(self, model_path: str, transforms=None, pooling_type="cls") -> None:
        """
        テンプレ
        :param model_path: モデルのパス
        :param transforms: textの前処理
        :param pooling_type: プーリングタイプ
        """
        self.model_path = model_path
        self.transforms = transforms
        self.pooling_type = pooling_type
        self.model = None
        self.max_length = None
        self.batch_size = None

    @abstractmethod
    def get_vector(self, sentence: str) -> np.array:
        """
        文章から分散表現を取得する
        :param sentence: 文章
        :return: 文書ベクトル
        """
        pass

    @abstractmethod
    def get_matrix(self, sentences: Union[List[str], pd.Series]) -> np.array:
        """
        文章群から分散表現を取得する
        :param sentences: 文章群
        :return: 文章ベクトル群
        """
        pass
