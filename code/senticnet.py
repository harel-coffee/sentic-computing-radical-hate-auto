"""
Set of functions that use the SenticNet resources

Author: Oscar Araque
"""

import numpy as np
import pandas as pd
from nltk import ngrams
import functools

import read_data
config = read_data.read_config()

def load_affectivespace():
    path = config['senticnet']['affectivespace']
    df = pd.read_csv(path, header=None)
    df = df[~df.isnull().any(axis=1)]
    vocab = set(df[0].values)
    d_tmp = df.set_index(0).to_dict('split')
    d = dict()
    for token, value in zip(d_tmp['index'], d_tmp['data']):
        d[token] = value
    return d, vocab


def extract_ngrams(texts, nrange=(1,1)):
    texts_ngrams = []
    ngram_indices = range(nrange[0], nrange[1]+1)
    for text in texts:
        text_ngrams = []
        for n_i in ngram_indices:
            t = list(ngrams(text.split(), n_i))
            text_ngrams.extend(['_'.join(gram) for gram in t])
        text_ngrams = set(text_ngrams)
        texts_ngrams.append(text_ngrams)
    return texts_ngrams


def extract_affectivespace_features(X, affectivespace, affectivespace_vocab, pooling=np.average):
    X_feats = []
    for x in X:
        coincidences = set(x) & affectivespace_vocab
        if len(coincidences) == 0:
            # if OOV, get the 1st term in vocabulary to know the feature size
            # and fill with zeroes
            X_feats.append([0]*(len(affectivespace['a_little'])))
            continue
        x_affectivespace = np.array([affectivespace[coin] for coin in coincidences])
        x_affectivespace = pooling(x_affectivespace, axis=0)
        X_feats.append(x_affectivespace)
    return np.array(X_feats)


def load_senticnet():
    path = config['senticnet']['senticnet']
    df = pd.read_pickle(path)
    vocab = set(df.index.values)

    # remove semantics information
    df.drop(columns=['semantics1', 'semantics2', 'semantics3', 'semantics4', 'semantics5'],
            inplace=True)

    # dummies for moods
    primary_moods = pd.get_dummies(df['primary_mood'])
    primary_moods.columns = ["primary" +  col for col in primary_moods.columns]
    df.drop(columns=['primary_mood'], inplace=True)
    secondary_moods = pd.get_dummies(df['secondary_mood'])
    secondary_moods.columns = ["secondary" +  col for col in secondary_moods.columns]
    df.drop(columns=['secondary_mood'], inplace=True)

    # transform polarity
    polarity_trans = lambda p: 1 if p == 'positive' else 0
    df['polarity_label'] = df['polarity_label'].apply(polarity_trans)

    # put together the dataframes
    df = pd.concat([df, primary_moods, secondary_moods], axis=1)
    for col in ('pleasantness_value', 'attention_value', 'sensitivity_value',
       'aptitude_value', 'polarity_label', 'polarity_value',):
        df[col] = pd.to_numeric(df[col])
    df.sort_index(axis=0, inplace=True)

    # transform to dictionary
    d_tmp = df.to_dict('split')
    d = dict()
    for token, value in zip(d_tmp['index'], d_tmp['data']):
        d[token] = value
    return d, vocab, df

def extract_senticnet_features(X, senticnet, senticnet_vocab):
    X_feats = []
    for x in X:
        coincidences = set(x) & senticnet_vocab
        if len(coincidences) == 0:
            # if OOV, get the 1st term in vocabulary to know the feature size
            # and fill with zeroes
            X_feats.append([0]*(len(senticnet['a_little'])*2))
            continue
        x_senticnet = np.array([senticnet[coin] for coin in coincidences])

        # pooling functions: mean and max
        pooled = np.concatenate((
            np.mean(x_senticnet, axis=0),
            np.max(x_senticnet, axis=0),
        ), axis=0)
        X_feats.append(pooled)
    return np.array(X_feats)
