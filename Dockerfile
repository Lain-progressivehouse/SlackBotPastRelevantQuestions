# kaggleのpython環境をベースにする
#FROM gcr.io/kaggle-images/python:v56
#FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update --fix-missing && \
    apt-get -y upgrade && \
    apt-get -y install \
        python3.7 \
        python3-pip \
        nano \
        wget \
        curl \
        git \
        unzip \
        sudo \
        zsh \
        gcc \
        g++ \
        make \
# Mecabのインストール
        mecab libmecab-dev mecab-ipadic mecab-ipadic-utf8

# HuggingfaceのTokenizersを動かすためにrustをinstall
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y && \
    export PATH="$HOME/.cargo/bin:$PATH"

COPY requirements.txt .

# ライブラリの追加インストール
RUN pip3 install -U pip && \
    pip install -r requirements.txt