import os
import re
import yaml
import pickle
import numpy as np
import pandas as pd
from gsitk.preprocess import pprocess_twitter, simple, Preprocesser
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin

import spacy
nlp = spacy.load("en_core_web_sm")

def read_config(path="config.yaml"):
    with open(path, 'r') as f:
        config = yaml.load(f)
    return config

class JoinTransformer(TransformerMixin):

    def transform(self, X, y=None, **fit_params):
        return [' '.join(x)for x in X]

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self

def tweets_per_user(df, key='text'):
    return df.groupby(level=0).count()[key]

def concatenate_all_tweets_per_user(df, key='text'):
    return df[key].groupby(level=0).apply(lambda t: ' '.join(t))

def remove_translation_annotation(df, key='text'):
    annotations = ['ENGLISH TRANSLATION: ', 'ENGLISH TRANSCRIPT: ']
    for anno in annotations:
        df[key] = df[key].str.replace(anno, '')
    return df

def preprocess_text(dataset):
    pp_pipe = Pipeline([
        ('twitter', Preprocesser(pprocess_twitter)),
        ('simple', Preprocesser(simple)),
        ('join', JoinTransformer()),
    ])

    dataset = remove_translation_annotation(dataset)
    dataset['text'] = pp_pipe.fit_transform(dataset['text'])
    return dataset


def read(pro_path, neutral_path):
    pro_isis = pd.read_csv(pro_path, escapechar="\\")
    pro_isis = pro_isis.set_index('username')
    
    neutral_isis = pd.read_csv(neutral_path, escapechar="\\", encoding='ISO-8859-2')
    neutral_isis = neutral_isis.set_index('snsuserid')
    
    # get all tweets of each user
    pro_tweets = concatenate_all_tweets_per_user(pro_isis)
    neu_tweets = concatenate_all_tweets_per_user(neutral_isis)
    
    # prepare dataset: {texts (all tweets per user), annotations (radical/non-radical)}
    users = np.concatenate([pro_tweets.index, neu_tweets.index], axis=0)
    texts = np.concatenate([pro_tweets.values, neu_tweets.values], axis=0)
    labels = np.concatenate([np.ones(pro_tweets.shape[0], dtype=int),
                             np.zeros(neu_tweets.shape[0], dtype=int)], axis=0)
    assert texts.shape[0] == labels.shape[0] # sanity check
    dataset = pd.DataFrame(data=np.vstack((texts, labels)).T, index=users, columns=['text', 'label'])
    dataset['label'] = dataset['label'].astype(int)

    preprocess_text(dataset)

    return dataset


def concatenate_all_tweets_per_user_566format(df, by='userid', key='text'):
    return df.groupby(by='userid')[key].apply(lambda t: ' '.join(t))

re_pattern = re.compile(u'[^\u0000-\uD7FF\uE000-\uFFFF]', re.UNICODE)

def filter_using_re(unicode_string):
    return re_pattern.sub(u'\uFFFD', unicode_string)

def read_file_line_format(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    dataset_ = []
    for line in lines:
        line = eval(line)
        userid = line[0]
        for tweet in line[1]:
            text = tweet[1].encode('utf-8', "backslashreplace").decode('utf-8')
#             text
            dataset_.append([userid, tweet[0], text, tweet[2], tweet[3]])
    dataset = pd.DataFrame(columns=['userid', 'tweetid', 'text', 'date', 'lang'], data=dataset_)
#     df['text'] = df['text'].apply(lambda t: t.encode('utf-8', "replace").decode('utf-8', 'replace'))
    return dataset
        



def read_pro_anti_566(pro_566_path, anti_566_path):
    pro_isis = read_file_line_format(pro_566_path)
    pro_isis = preprocess_text(pro_isis)
    pro_tweets = concatenate_all_tweets_per_user_566format(pro_isis)

    anti_isis = read_file_line_format(anti_566_path)
    anti_isis = preprocess_text(anti_isis)
    anti_tweets = concatenate_all_tweets_per_user_566format(anti_isis)

    users = np.concatenate([pro_tweets.index, anti_tweets.index], axis=0)
    texts = np.concatenate([pro_tweets.values, anti_tweets.values], axis=0)
    labels = np.concatenate([np.ones(pro_tweets.shape[0], dtype=int),
                             np.zeros(anti_tweets.shape[0], dtype=int)], axis=0)
    assert texts.shape[0] == labels.shape[0] # sanity check
    dataset = pd.DataFrame(data=np.vstack((texts, labels)).T, index=users, columns=['text', 'label'])
    dataset['label'] = dataset['label'].astype(int)
    
    return dataset


def read_magazines_corpus(magazines_path):
    dataset = pd.read_json(magazines_path)
    dataset['text'] = dataset['articleBody']

    preprocess_text(dataset)

    return dataset

def read_semeval19_hatespeech_dataset(data_path):
    train = pd.read_csv(os.path.join(data_path, "train_en_A.tsv"), sep='\t')
    dev = pd.read_csv(os.path.join(data_path, "dev_en_A.tsv"), sep='\t')
    dataset = pd.concat((train, dev), axis=0)
    dataset['label'] = dataset['HS'].values.astype(int)

    preprocess_text(dataset)

    return dataset


def read_davidson_dataset(data_path):
    df = pd.read_csv(data_path)
    df.rename(columns={'class': 'label', 'tweet': 'text'}, inplace=True)

    preprocess_text(df)

    return df


def lemmatize_text(text):
    return ' '.join([token.lemma_.lower() for token in  nlp.tokenizer(text)])


def read_all_datasets(cached_data="../data/all_datasets.pck"):
    if os.path.exists(cached_data):
        with open(cached_data, 'rb') as f:
            data = pickle.load(f)
        return data

    print("No cache, loading from source...")

    config = read_config()['data']
    data = dict()
    data['pro-neu'] = read(config['pro_path'], config['neutral_path'])
    data['pro-anti'] = read_pro_anti_566(config['pro_566_path'], config['anti_566_path'])
    data['magazines'] = read_magazines_corpus(config['magazines'])
    data['semeval19hate'] = read_semeval19_hatespeech_dataset(config['semeval19hate'])
    data['davidson'] = read_davidson_dataset(config['davidson'])

    for data_name in data.keys():
        data[data_name]['text'] = data[data_name]['text'].apply(lemmatize_text)

    with open(cached_data, 'wb') as f:
        pickle.dump(data, f)

    return data

