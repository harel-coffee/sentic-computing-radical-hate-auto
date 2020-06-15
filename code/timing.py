"""
Timing the different methods.

Author: Oscar Araque
"""

import pickle
import timeit
import functools
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer

import stats
import senticnet
import features
import read_data

def set_timestamp():
    return timeit.default_timer()

def time_elapsed(start, end):
    return round(end - start, 1)

times = dict()
times['load'] = dict()

print("Reading datasets...")
data = read_data.read_all_datasets()
print("Done")

print("Loading SenticNet resources")
start = set_timestamp()
affectivespace, affectivespace_vocab = senticnet.load_affectivespace()
end = set_timestamp()
times['load']['affectivespace'] = time_elapsed(start, end)


start = set_timestamp()
senticnet5, senticnet5_vocab, senticnet5_full = senticnet.load_senticnet()
end = set_timestamp()
times['load']['senticnet5'] = time_elapsed(start, end)
print("Done")

print("Loading embeddings")
start = set_timestamp()
emb = features.load_embeddings()
end = set_timestamp()
times['load']['embeddings'] = time_elapsed(start, end)
print("Done")

print("Extracting pre-features")
times['prefeats'] = dict()

times['prefeats']['ngrams'] = dict()
ngrams = dict()
for data_name in tqdm(data.keys()):
    start = set_timestamp()
    ngrams[data_name] = senticnet.extract_ngrams(data[data_name]['text'], (1,5))
    end = set_timestamp()
    times['prefeats']['ngrams'][data_name]  = time_elapsed(start, end)

times['prefeats']['simonlex'] = dict()
custom_lexicon = dict()
for data_name in tqdm(data.keys()):
    start = set_timestamp()
    custom_lexicon[data_name] = features.generate_custom_lexicon(data[data_name])
    end = set_timestamp()
    times['prefeats']['simonlex'][data_name]  = time_elapsed(start, end)


times['prefeats']['simonsenticnet'] = dict()
senticnet5_lexicon = dict()
for data_name in tqdm(data.keys()):
    start = set_timestamp()
    senticnet5_lexicon[data_name] = features.generate_senticnet5_lexicon(data[data_name], senticnet5_full)
    end = set_timestamp()
    times['prefeats']['simonsenticnet'][data_name]  = time_elapsed(start, end)
print("Done!")

print("Extracting features")
times['feats'] = dict()
times['feats']['affectivespace'] = dict()
# affectivespace
for data_name in tqdm(data.keys()):
    start = set_timestamp()
    senticnet.extract_affectivespace_features(
        ngrams[data_name],
        affectivespace=affectivespace,
        affectivespace_vocab= affectivespace_vocab,
    )
    end = set_timestamp()
    times['feats']['affectivespace'][data_name]  = time_elapsed(start, end)

# senticnet5
times['feats']['senticnet5'] = dict()
for data_name in tqdm(data.keys()):
    start = set_timestamp()
    senticnet.extract_senticnet_features(
        ngrams[data_name],
        senticnet=senticnet5,
        senticnet_vocab= senticnet5_vocab,
    )
    end = set_timestamp()
    times['feats']['senticnet5'][data_name]  = time_elapsed(start, end)


# TF-IDF
times['feats']['tfidf'] = dict()
for data_name in tqdm(data.keys()):
    start = set_timestamp()
    tfidf = TfidfVectorizer(ngram_range=(1,1), max_features=5000)
    tfidf.fit_transform(data[data_name]['text'].values)
    end = set_timestamp()
    times['feats']['tfidf'][data_name]  = time_elapsed(start, end)

# simon - custom lexicon
times['feats']['simon'] = dict()
for data_name in tqdm(data.keys()):
    start = set_timestamp()
    features.simon_feat_extractor(
        dataset=data[data_name],
        lexicon=custom_lexicon[data_name],
        embedding_model=emb,
        n_lexicon_words=200,
        percentage=100,
    )
    end = set_timestamp()
    times['feats']['simon'][data_name]  = time_elapsed(start, end)

# simon - sentincnet5 lexicon
times['feats']['simonsenticnet'] = dict()
for data_name in tqdm(data.keys()):
    start = set_timestamp()
    features.simon_feat_extractor(
        dataset=data[data_name],
        lexicon=senticnet5_lexicon[data_name],
        embedding_model=emb,
        n_lexicon_words=200,
        percentage=100,
    )
    end = set_timestamp()
    times['feats']['simonsenticnet'][data_name]  = time_elapsed(start, end)

print("DONE")


print("Saving results...")
with open("../results/timing.pck", 'wb') as f:
    pickle.dump(times, f)
print("Done")

print("Finished, have a good day!")
