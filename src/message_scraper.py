import os
import time

from slack import WebClient
from collections import defaultdict
import pandas as pd
import re
from datetime import datetime, timedelta


class MessageScraper(object):
    """
    Slackのチャンネル上の質問データを取得するクラス
    """

    def __init__(self, cfg):
        """
        :param cfg: コンフィグ(現状はslackのトークンの情報のみ)
        """
        self.token = cfg["token"]
        self.client = WebClient(token=self.token)

    @staticmethod
    def _messages_parser(message: dict):
        """
        slackのメッセージのパーサー
        :param message:
        :return:
        """
        if not message:
            return None
        if message.get("user") != "W01AVA2GS3S":
            return None
        if len(message.get("blocks", {})) != 3:
            return None

        line = []
        user = re.match(r"<@[\w\W]*?>", message.get("text")).group()[2:-1]
        line.append(("user", user))
        line.append(("thread_ts", message.get("thread_ts")))
        line.append(("reply_count", message.get("reply_count")))
        text = message.get("blocks")[1].get("text").get("text")
        question_category = re.match(r"\[[\W\w]*?\]", text)
        if question_category:
            line.append(("question_category", question_category.group()))
            line.append(("text", text[len(question_category.group()) + 1:]))
        else:
            line.append(("question_category", None))
            line.append(("text", text))

        return line

    def get_messages(self, channel_id: str, start_ts: int = None, days: int = None) -> pd.DataFrame:
        """
        過去の質問データを取得する
        :param channel_id: 取得するデータのチャンネル
        :param start_ts: 取得するデータの開始点(Noneの場合最新から)
        :param days: 開始点から過去何日分のデータを取得するか(Noneの場合全て取得)
        :return: データフレーム
        """
        if not start_ts:
            start_ts = datetime.now()
        else:
            start_ts = datetime.fromtimestamp(start_ts)
        oldest = (start_ts - timedelta(days=days)).timestamp() if days else None
        start_ts = start_ts.timestamp()
        dd = defaultdict(list)
        cursor = None
        while True:
            time.sleep(0.3)
            response = self.client.conversations_history(
                channel=channel_id,
                limit=100,
                latest=start_ts,
                oldest=oldest,
                cursor=cursor,
                inclusive=1
            )
            for lines in [self._messages_parser(message) for message in response.get("messages", {})]:
                if lines is None:
                    continue
                for key, value in lines:
                    dd[key].append(value)
            cursor = response.get("response_metadata", {}).get("next_cursor")
            if cursor is None:
                break

        df = pd.DataFrame(dd)
        df["channel"] = channel_id
        return df

    def get_replies(self, channel_id: str, message_ts: int) -> str:
        """
        指定した質問の回答データを取得する
        :param channel_id: 取得する質問のチャンネル
        :param message_ts: その質問のthread_ts
        :return: 回答
        """
        messages = []
        cursor = None
        while True:
            time.sleep(0.01)
            try:
                response = self.client.conversations_replies(
                    channel=channel_id,
                    ts=message_ts,
                    cursor=cursor
                )
                for message in response.get("messages"):
                    messages.append(message.get("text"))
                cursor = response.get("response_metadata", {}).get("next_cursor")
            except:
                pass
            if cursor is None:
                break
        return messages[2: -1]
