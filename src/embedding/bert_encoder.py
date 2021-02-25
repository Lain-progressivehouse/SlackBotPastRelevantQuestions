import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from typing import Union, List
from src.embedding.encoder import Encoder


class BertDataset(Dataset):
    def __init__(self,
                 sentences: Union[List[str], pd.Series],
                 tokenizer: AutoTokenizer,
                 max_length=128,
                 transforms=None):
        self.max_length = max_length
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.transforms = transforms

    def __getitem__(self, index):
        data = {}
        sentence = self.sentences[index]
        sentence = str(sentence)
        if self.transforms is not None:
            sentence = self.transforms(sentence)
        input_ids, attention_mask = self._tokenize(sentence)
        data["input_ids"] = torch.tensor(input_ids)
        data["attention_mask"] = torch.tensor(attention_mask)
        return data

    def __len__(self):
        return len(self.sentences)

    def _tokenize(self, sentence):
        id_dict = self.tokenizer.encode_plus(sentence,
                                             max_length=self.max_length,
                                             pad_to_max_length=True,
                                             truncation=True)
        return id_dict["input_ids"], id_dict["attention_mask"]


class BertEncoder(Encoder):

    def __init__(self,
                 model_path: str,
                 transforms=None,
                 pooling_type="cls",
                 max_length=128,
                 batch_size=32,
                 disable_tqdm=True) -> None:
        """
        BERT
        :param model_path: モデルのパス
        :param transforms: textの前処理
        :param pooling_type: poolingの種類
        :param max_length: トークンの最大長
        :param batch_size: バッチサイズ
        :param disable_tqdm: 推論時にプログレスバーを表示するかどうか
        """
        pooling_list = ["cls", "mean", "max", "concat"]
        assert pooling_type in pooling_list, f"{pooling_type} is a non-existent pooling_type {pooling_list}"
        super().__init__(model_path, transforms=transforms, pooling_type=pooling_type)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length
        self.batch_size = batch_size
        self.disable_tqdm = disable_tqdm

    def get_vector(self, sentence: str) -> np.array:
        if self.transforms is not None:
            sentence = self.transforms(sentence)
        token = self.tokenizer.encode_plus(
            sentence,
            max_length=self.max_length,
            pad_to_max_length=True,
            truncation=True
        )
        with torch.no_grad():
            _, _, outputs = self.model(
                input_ids=torch.tensor([token["input_ids"]]).to(self.device),
                attention_mask=torch.tensor([token["attention_mask"]]).to(self.device),
                output_hidden_states=True
            )

        # output = torch.mean(torch.stack([outputs[-i] for i in range(1, 3)]), 0)
        output = outputs[-1]

        vector = output[:, 0]

        if self.pooling_type == "cls":
            vector = output[:, 0]
        elif self.pooling_type == "mean":
            vector = output.mean(axis=1)
        elif self.pooling_type == "max":
            vector = output.max(axis=1)[0]
        elif self.pooling_type == "concat":
            vector = torch.cat([output.mean(axis=1), output[:, 0]], axis=1)

        vector = vector.to(torch.device("cpu"))
        return vector.squeeze().detach().clone().numpy()

    def get_matrix(self, sentences: Union[List[str], pd.Series]) -> np.array:
        data_set = BertDataset(sentences, self.tokenizer, max_length=self.max_length, transforms=self.transforms)
        data_loader = DataLoader(data_set, batch_size=self.batch_size, shuffle=False)
        preds = []
        for batch in tqdm(data_loader, disable=self.disable_tqdm):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
            with torch.no_grad():
                _, _, outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                # output = torch.mean(torch.stack([outputs[-i] for i in range(1, 3)]), 0)
                output = outputs[-1]

                if self.pooling_type == "cls":
                    output = output[:, 0]
                elif self.pooling_type == "mean":
                    output = output.mean(axis=1)
                elif self.pooling_type == "max":
                    output = output.max(axis=1)[0]
                elif self.pooling_type == "concat":
                    output = torch.cat([output.mean(axis=1), output[:, 0]], axis=1)

                output = output.to(torch.device("cpu"))
                preds.append(output.detach().clone().numpy())
        return np.concatenate(preds, axis=0)
