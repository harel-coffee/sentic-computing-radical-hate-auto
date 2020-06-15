import string
from collections import Counter
from itertools import chain
from nltk.corpus import stopwords
from gsitk.features import simon
from gsitk.features.simon import simon_pipeline
from gensim.models import KeyedVectors

import read_data
config = read_data.read_config()


# define some filterings words
filter_words = set(stopwords.words('english')) | set(string.punctuation)

def load_embeddings(model='facebook'):
    emb_path = config['data']['embeddings']
    if model == 'facebook':
        embbeddings = KeyedVectors.load_word2vec_format(emb_path, binary=False)
    elif model == 'google':
        embbeddings = KeyedVectors.load_word2vec_format(emb_path, binary=True)
    elif model == 'glove':
        embbeddings = KeyedVectors.load_word2vec_format(emb_path, binary=False)
    else:
        raise ValueError
    return embbeddings

def simon_feat_extractor(dataset, lexicon, embedding_model, n_lexicon_words=200, percentage=100):

    simon_sentiment = simon.Simon(lexicon=lexicon, n_lexicon_words=n_lexicon_words,
                                  embedding=embedding_model)
    simon_sentiment_full = simon_pipeline(simon_sentiment, percentage)

    sim_feats = simon_sentiment_full.fit_transform(dataset['text'].str.split(' ').values, dataset['label'])
    return sim_feats


def generate_custom_lexicon(dataset):
    counter = Counter(chain.from_iterable(dataset['text'].str.split(' ').values))
    selection = sorted([(word, count) for word, count in counter.items()], key=lambda wc: wc[1], reverse=True)
    selection = [word for word, _ in selection if word not in filter_words]
    selection = [selection]
    return selection


def generate_senticnet5_lexicon(dataset, senticnet5_full, polarity_threshold=0.8):
    sn_noindx = senticnet5_full.reset_index(0)
    # absolute value of polarities
    sn_noindx['polarity_value'] = sn_noindx['polarity_value'].abs()
    # we get only unigrams, and filter with the polarity threshold
    sn = sn_noindx[(sn_noindx['index'].str.split("_").apply(len) == 1) & (sn_noindx['polarity_value'] > polarity_threshold)]
    sn = sn[['index', 'polarity_value']]

    # count the appearances of the words in the dataset
    counter = Counter(chain.from_iterable(dataset['text'].str.split(' ').values))
    sn['freq'] = sn['index'].apply(lambda w: counter[w])

    # filter the filter words
    sn = sn[~sn['index'].isin(filter_words)]
    # order by frequency and polarity value
    lexicon_list = sn.sort_values(['freq', 'polarity_value'], ascending=False)['index'].values

    return [lexicon_list]
