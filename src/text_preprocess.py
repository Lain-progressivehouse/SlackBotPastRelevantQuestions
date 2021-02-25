"""
todo: __lt__や__gt__を実装してclassのソートをできるようにする
"""

import re
import emoji
import mojimoji
from typing import List
from sudachipy import tokenizer
from sudachipy import dictionary


class Compose(object):
    """
    前処理クラスのCompose
    """

    def __init__(self, transforms: List[object]):
        self.transforms = transforms

    def __call__(self, text: str):
        text = str(text)
        for t in self.transforms:
            text = t(text)
        return text


class ReplyRemove(object):
    """
    リプライの削除(先頭matchのみ)
    """

    def __init__(self):
        self.pattern = re.compile(r"@[\w]* ")

    def __call__(self, text: str) -> str:
        text = text.strip()
        while self.pattern.match(text):
            text = text[self.pattern.match(text).end():].strip()
        return text


class URLRemove(object):
    """
    URLを削除または変換
    """

    def __init__(self, replace=""):
        self.replace = replace
        self.pattern = re.compile(r"https?://[\w!?/+\-_~;.,*&@#$%()'[\]]+")

    def __call__(self, text: str) -> str:
        return self.pattern.sub(self.replace, text)


class NumberRemove(object):
    """
    数字を削除または変換
    """

    def __init__(self, replace="0"):
        self.replace = replace
        self.pattern = re.compile(r"[0-9０-９]+")

    def __call__(self, text: str) -> str:
        return self.pattern.sub(self.replace, text)


class SpaceRemove(object):
    """
    全角スペースやタブを半角スペースに変換
    複数の連続した半角スペースをひとつだけに変換する
    """

    def __init__(self, replace=" "):
        self.replace = replace
        self.pattern = re.compile(r"\s[\s]*")

    def __call__(self, text: str) -> str:
        return self.pattern.sub(self.replace, text).strip()


class LineRemove(object):
    """
    全角スペースやタブを半角スペースに変換
    複数の連続した半角スペースをひとつだけに変換する
    """

    def __init__(self, replace=" "):
        self.replace = replace
        self.pattern = re.compile(r"\n")

    def __call__(self, text: str) -> str:
        return self.pattern.sub(self.replace, text).strip()


class SymbolRemove(object):
    """
    記号を削除または変換
    """

    def __init__(self, replace=" "):
        self.replace = replace
        self.pattern = re.compile(
            r"[-~︰-＠．◆©～－｜‘’・〔〕▼！＠＃＄％＾＆\-|\\＊\“（）＿■×※⇒—●(：〜＋=)／*&^%$#@!~`){}…\[\]\"\'\”:;<>?＜＞？､、。・･,./『』【】「」→←○¥]+")

    def __call__(self, text: str) -> str:
        return self.pattern.sub(self.replace, text)


class EmojiRemove(object):
    """
    絵文字を削除または変換
    """

    def __init__(self, replace=" "):
        self.replace = replace
        self.pattern = re.compile(rf"[{''.join(emoji.UNICODE_EMOJI)} ]+")

    def __call__(self, text: str) -> str:
        return self.pattern.sub(self.replace, text).strip()


class ZenToHan(object):
    """
    全角を半角に変換
    """

    def __call__(self, text: str) -> str:
        return mojimoji.zen_to_han(text, kana=False)


class Tokenizer(object):
    """
    Sudachiによる単語の分割
    """

    def __init__(self, hinshi_list: List[str] = None, split_mode: str = "C"):
        """
        :param hinshi_list: 使用する品詞のリスト. example) hinshi_list=["動詞", "名詞", "形容詞"]
        :param split_mode:
        """
        split_mode_list = ["A", "B", "C"]
        assert split_mode in split_mode, f"{split_mode} is a non-existent split_mode {split_mode_list}"
        split_dic = {
            "A": tokenizer.Tokenizer.SplitMode.A,
            "B": tokenizer.Tokenizer.SplitMode.B,
            "C": tokenizer.Tokenizer.SplitMode.C,
        }
        self.tokenizer_obj = dictionary.Dictionary().create()
        self.mode = split_dic[split_mode]
        self.hinshi_list = hinshi_list

    def __call__(self, text: str) -> str:
        if self.hinshi_list:
            return " ".join([m.normalized_form() for m in self.tokenizer_obj.tokenize(text, self.mode) if
                             m.part_of_speech()[0] in self.hinshi_list and m.normalized_form() != " "])
        return " ".join(
            m.normalized_form() for m in self.tokenizer_obj.tokenize(text, self.mode) if m.normalized_form() != " ")


class BasicPreprocessor(object):
    def __init__(self):
        self.transform = Compose(
            [
                # ReplyRemove(),
                # URLRemove(),
                ZenToHan(),
                # NumberRemove(),
                # SymbolRemove(),
                # EmojiRemove(),
                SpaceRemove(),
                # Tokenizer(hinshi_list=["動詞", "名詞", "形容詞"])
            ]
        )

    def __call__(self, text: str, wakati=True):
        if wakati:
            return self.transform(text)
        else:
            return self.transform(text).split()
