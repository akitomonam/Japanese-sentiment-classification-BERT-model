# %% [markdown]
# # 発話感情分類モデルの作成

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# %% [markdown]
# # データの準備

# %% [markdown]
# ## データダウンロード

# %%
# !rm - r . / data

# %%
# !git clone https: // github.com / ids - cv / wrime.git . / data/

# %%
# !rm - rf . / data / .git

# %%
# !wget - P . / data / https: // www.dropbox.com / s / rkzyeu58p48ndz3 / japanese_empathetic_dialogues.xlsx

# %% [markdown]
# ## データ前処理

# %% [markdown]
# ## japanese_empathetic_dialogues前処理

# %%
df_env = pd.read_excel(
    "./data/japanese_empathetic_dialogues.xlsx", sheet_name="状況文")
df_env = df_env.rename(columns={'作業No': 'ID'})

# %%
df_utt = pd.read_excel(
    "./data/japanese_empathetic_dialogues.xlsx", sheet_name="対話")

# %%
df = pd.merge(df_utt, df_env, how="left", on="ID")

# %%
empathtic_names = df["感情"].unique()
print(empathtic_names)
print(len(empathtic_names))
print(df["感情"].value_counts())

# %%
# 話者がAだけ残す
df = df[df["話者"] == "A"]

# %%
df.drop(["ID", "話者", "状況文"], axis=1, inplace=True)
df = df.reindex(columns=['発話', '感情'])
df[50:85]

# %%
# 32個の感情を8個にまとめる
emotions_dict = {
    "喜び": ["感謝する", "感動する", "楽しい", "満足"],
    "悲しみ": ["悲しい", "さびしい", "がっかりする", "打ちのめされる", "感傷的になる"],
    "期待": ["わくわくする", "期待する", "待ち望む"],
    "驚き": ["おどろく"],
    "怒り": ["怒る", "いらいらする", "激怒する"],
    "恐れ": ["怖い", "恐ろしい", "不安に思う", "懸念する"],
    "嫌悪": ["うしろめたい", "嫌悪感を抱く", "恥ずかしい", "恥じる"],
    "信頼": ["自信がある", "信頼する", "誠実な気持ち"]
}
drop_emotions = ["誇りに思う", "心構えする", "羨ましい", "懐かしい", "思いやりを持つ"]

# %%
for p_emo, c_emos in emotions_dict.items():
    df["感情"].replace(c_emos, p_emo, inplace=True)

# %%
# 使用しない感情の行を削除する
for emo in drop_emotions:
    df = df[df["感情"] != emo]

# %%
empathtic_names = df["感情"].unique()
print(empathtic_names)
print(len(empathtic_names))
print(df["感情"].value_counts())

# %% [markdown]
# ## WRIME前処理

# %%
df_wrime = pd.read_csv('./data/wrime-ver2.tsv', delimiter='\t')
# 必要な列だけ抽出
df_wrime = df_wrime.loc[:, ["Sentence", "Train/Dev/Test",
                            "Writer_Joy", "Writer_Sadness", "Writer_Anticipation", "Writer_Surprise", "Writer_Anger", "Writer_Fear", "Writer_Disgust", "Writer_Trust", "Writer_Sentiment",
                            "Avg. Readers_Joy", "Avg. Readers_Sadness", "Avg. Readers_Anticipation", "Avg. Readers_Surprise", "Avg. Readers_Anger", "Avg. Readers_Fear",
                            "Avg. Readers_Disgust", "Avg. Readers_Trust", "Avg. Readers_Sentiment"]]
len(df_wrime.columns)

# %%
# 2人のアノテーターの合計の多数決で感情を決定
# df_wrime.isnull().sum()
add_emotion_dict = {
    "喜び": ["Writer_Joy", "Avg. Readers_Joy"],
    "悲しみ": ["Writer_Sadness", "Avg. Readers_Sadness"],
    "期待": ["Writer_Anticipation", "Avg. Readers_Anticipation"],
    "驚き": ["Writer_Surprise", "Avg. Readers_Surprise"],
    "怒り": ["Writer_Anger", "Avg. Readers_Anger"],
    "恐れ": ["Writer_Fear", "Avg. Readers_Fear"],
    "嫌悪": ["Writer_Disgust", "Avg. Readers_Disgust"],
    "信頼": ["Writer_Trust", "Avg. Readers_Trust"]
    # "Sentiment":["Writer_Sentiment","Avg. Readers_Sentiment"] （今回は感情極性を使わない）
}

# %%
# それぞれの感情で合計値を計算
for emo_p, emo_c_list in add_emotion_dict.items():
    df_wrime = pd.concat([df_wrime, pd.DataFrame(
        df_wrime.loc[:, emo_c_list].sum(axis=1), columns=[emo_p])], axis=1)

# %%
df_wrime = df_wrime.loc[:, ["Sentence"] +
                        list(add_emotion_dict.keys())]  # 必要な列だけ抽出
df_wrime.rename(columns={"Sentence": "発話"}, inplace=True)  # 列名変更

# %%
# 感情ラベルを数値で多数決してラベルを決定
df_wrime = pd.concat([df_wrime, pd.DataFrame(df_wrime.loc[:, list(add_emotion_dict.keys(
))].idxmax(axis=1), columns=["感情"])], axis=1)  # 各感情で一番大きい感情を取り出し新しく感情ラベル列を作成

# %%
df_wrime = df_wrime.loc[:, ["発話", "感情"]]  # 使用する列だけ抽出
df_wrime

# %%
print(len(df_wrime["感情"].unique()))
print(df_wrime["感情"].value_counts())

# %% [markdown]
# ## データフレームの結合

# %%
print(len(df))
print(len(df_wrime))
df = pd.concat([df, df_wrime], axis=0)
print(len(df))

# %%
# シャッフルする
df = df.sample(frac=1, random_state=0, ignore_index=True)

# %%
print(len(df["感情"].unique()))
print(df["感情"].value_counts())

# %%
df.sample(frac=1)

# %%
emotion_list = ['喜び', '悲しみ', '期待', '驚き', '怒り', '恐れ', '嫌悪', '信頼']

# %%
# label to num
df["感情"].replace(emotion_list, range(len(emotion_list)), inplace=True)

# %%
df.sample(frac=1)

# %% [markdown]
# ## 発話テキスト前処理

# %%
import re

# %%


def text_preprocessing(text):
    # 「]の削除
    text = text.replace('「', '')
    text = text.replace('」', '')
    # URLの削除
    text = re.sub(r'http?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', text)
    text = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', text)
    # pic.twitter.comXXXの削除
    text = re.sub(r'pic.twitter.com/[\w/:%#\$&\?\(\)~\.=\+\-]+', '', text)
    # 全角記号削除
    text = re.sub(
        "[\uFF01-\uFF0F\uFF1A-\uFF20\uFF3B-\uFF40\uFF5B-\uFF65\u3000-\u303F]", '', text)
    # 半角記号の置換
    text = re.sub(r'[!-/:-@[-`{-~]', r' ', text)
    # 全角記号の置換 (ここでは0x25A0 - 0x266Fのブロックのみを除去)
    text = re.sub(u'[■-♯]', ' ', text)
    # 数値をすべて0に変換
    text = re.sub(r'\d+', '0', text)
    text = text.replace("\n", "")
    text = text.replace("。", "")
    text = text.replace(".", "")
    text = text.replace(",", "")
    text = text.lower()
    return text


# %%
df["発話"] = df["発話"].map(text_preprocessing)

# %%
df.sample(frac=1)

# %% [markdown]
# ## データの分割

# %%
from sklearn.model_selection import train_test_split

# %%
df = df.rename(columns={'感情': 'label'})
df = df.rename(columns={'発話': 'text'})

# %%
data_train, data_test = train_test_split(
    df, random_state=111, stratify=df.label)  # 訓練用とテスト用に分割 defalut 25%がテストデータ
print(len(data_train))
print(data_train["label"].value_counts() / len(data_train))
print(len(data_test))
print(data_test["label"].value_counts() / len(data_test))

# %%
train_docs = data_train["text"].tolist()
train_labels = data_train["label"].tolist()
len(train_docs)

# %%
test_docs = data_test["text"].tolist()
test_labels = data_test["label"].tolist()
len(test_docs)

# %% [markdown]
# # モデル構築

# %%
# GPU が利用できる場合は GPU を利用する

device = "cuda:0" if torch.cuda.is_available() else "cpu"
device

# %%
from transformers import BertForSequenceClassification, BertJapaneseTokenizer
from transformers import AdamW

sc_model = BertForSequenceClassification.from_pretrained(
    "cl-tohoku/bert-base-japanese-v2", num_labels=len(empathtic_names))
model = sc_model.to(device)
tokenizer = BertJapaneseTokenizer.from_pretrained(
    "cl-tohoku/bert-base-japanese-v2")

# %% [markdown]
# # エンコーディング
# Transformerモデルの入力としてはテンソル形式に変換する必要がある。
# 返り値のテンソルのタイプを選ぶことができる。ここではPyTorchのテンソル型で返してくれるよう、return_tensors='pt'としている。
#

# %%
train_encodings = tokenizer(train_docs, return_tensors='pt',
                            padding=True, truncation=True, max_length=128).to(device)
test_encodings = tokenizer(test_docs, return_tensors='pt',
                           padding=True, truncation=True, max_length=128).to(device)

# %%
import torch


class JpSentiDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = JpSentiDataset(train_encodings, train_labels)
test_dataset = JpSentiDataset(test_encodings, test_labels)

# %% [markdown]
# ## 評価関数の設定

# %%
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# %% [markdown]
# ## トレーニング


# %%
# !mkdir . / logs
# !ls

# %%
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=8,              # total number of training epochs
    per_device_train_batch_size=64,  # batch size per device during training
    per_device_eval_batch_size=128,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    # limit the total amount of checkpoints. Deletes the older checkpoints.
    save_total_limit=1,
    # Whether you want to pin memory in data loaders or not. Will default to True
    dataloader_pin_memory=False,
    # evaluation_strategy="epoch",     # Evaluation is done at the end of each epoch.
    evaluation_strategy="steps",
    logging_steps=50,
    logging_dir='./logs'
)

trainer = Trainer(
    # the instantiated 🤗 Transformers model to be trained
    model=model,
    args=training_args,                  # training arguments, defined above
    tokenizer=tokenizer,
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,             # evaluation dataset
    # The function that will be used to compute metrics at evaluation
    compute_metrics=compute_metrics
)

trainer.train()

# %% [markdown]
# ## 評価

# %%
trainer.evaluate(eval_dataset=test_dataset)

# %% [markdown]
# ## モデルの保存

# %%
save_directory = "./JEmpatheticDialogues_WRIME_model_not_same_labels_num"

tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

# %% [markdown]
# ## 学習グラフ

# %%
# %load_ext tensorboard
# %tensorboard - -logdir logs - -host localhost - -port 8888

# %% [markdown]
# # 推論テスト

# %%
# GPU が利用できる場合は GPU を利用する

device = "cuda:0" if torch.cuda.is_available() else "cpu"
device

# %%
from transformers import BertForSequenceClassification, BertJapaneseTokenizer
# 保存したモデルの読み込み
save_directory = "./JEmpatheticDialogues_WRIME_model_not_same_labels_num"
sc_model = BertForSequenceClassification.from_pretrained(save_directory)
model = sc_model.to(device)
tokenizer = BertJapaneseTokenizer.from_pretrained(save_directory)

# %%
from transformers import pipeline

sentiment_analyzer = pipeline(
    "sentiment-analysis", model=model.to("cpu"), tokenizer=tokenizer)

# %%
result = sentiment_analyzer("バーゲンだから買い物してくるよ")
print(list(add_emotion_dict.keys())[
      int(result[0]["label"].replace("LABEL_", ""))])
