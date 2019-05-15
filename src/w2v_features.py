import pandas as pd
import numpy as np
import gc
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from gensim import corpora, models
import spacy
import config

import daiquiri,logging
daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger(__name__)

logger.info("Reading in train and test dataframes...")
data_dir = config.DATA_DIR
train = pickle.load( open( data_dir+"train_features.p", "rb" ) )
test = pickle.load( open( data_dir+"test_features.p", "rb" ) )

train = train[['id','cleaned_tweet']]
test = test[['id','cleaned_tweet']]

# Get wor2vec features
logger.info("Creating w2v features using spacy w2v model...")
nlp = spacy.load('en_core_web_sm')

train_vectors = []
test_vectors = []

for tweet in train.cleaned_tweet.values:
    train_vectors.append(nlp(tweet).vector)
for tweet in test.cleaned_tweet.values:
    test_vectors.append(nlp(tweet).vector)

logger.info("Generating train and test w2v dataframe...")
train_vectors_df = pd.DataFrame(train_vectors)
test_vectors_df = pd.DataFrame(test_vectors)
train_vectors_df.columns = ["f_"+str(col) for col in train_vectors_df.columns]
test_vectors_df.columns = ["f_"+str(col) for col in test_vectors_df.columns]

logger.info("Dumping train and test word2vec features...")
pickle.dump( train_vectors_df, open( data_dir+"train_word2vec.p", "wb" ) )
pickle.dump( test_vectors_df, open( data_dir+"test_word2vec.p", "wb" ) )

del train_vectors_df,test_vectors_df,train_vectors,test_vectors
gc.collect()