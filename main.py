from src.metrics import get_predict_time
from src.embedding.swem_encoder import SwemEncoder
from src.embedding.bert_encoder import BertEncoder
from src.text_preprocess import (Compose, ZenToHan, SpaceRemove, NumberRemove)

from src import metrics

if __name__ == '__main__':
    swem_transforms = Compose([ZenToHan(), NumberRemove(), SpaceRemove()])
    swem_encoder = SwemEncoder(transforms=swem_transforms,
                               model_path="./data/model/jawiki.word_vectors.300d.bin")
    bert_transforms = Compose([ZenToHan(), SpaceRemove()])
    bert_encoder = BertEncoder(transforms=bert_transforms,
                               model_path="cl-tohoku/bert-base-japanese-whole-word-masking")
    sbert_encoder = BertEncoder(transforms=bert_transforms,
                                model_path="./data/model/training_bert_japanese/0_BERTJapanese/")

    metrics.metric_ensemble([swem_encoder, bert_encoder, sbert_encoder])