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

# Get the tfidf vectors #
logger.info("Creating TFIDF model with trigrams...")
tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,3), max_features=300)
tfidf_vec.fit_transform(train['cleaned_tweet'].values.tolist() + test['cleaned_tweet'].values.tolist())
train_tfidf = tfidf_vec.transform(train['cleaned_tweet'].values.tolist())
test_tfidf = tfidf_vec.transform(test['cleaned_tweet'].values.tolist())
train_tfidf_df = pd.DataFrame(train_tfidf.toarray(), columns=tfidf_vec.get_feature_names())
test_tfidf_df = pd.DataFrame(test_tfidf.toarray(), columns=tfidf_vec.get_feature_names())

logger.info("Dumping train and test TFIDF vectors for future modeling...")
pickle.dump( train_tfidf_df, open( data_dir+"train_tfidf.p", "wb" ) )
pickle.dump( test_tfidf_df, open( data_dir+"test_tfidf.p", "wb" ) )

del train_tfidf,test_tfidf,train_tfidf_df,test_tfidf_df
gc.collect()

# Get the tfidf vectors #
logger.info("Creating TFIDF char model with trigrams...")
tfidf_vec = TfidfVectorizer(analyzer='char', ngram_range=(1, 5), min_df=500, sublinear_tf=True,max_features=300)
tfidf_vec.fit_transform(train['cleaned_tweet'].values.tolist() + test['cleaned_tweet'].values.tolist())
train_tfidf = tfidf_vec.transform(train['cleaned_tweet'].values.tolist())
test_tfidf = tfidf_vec.transform(test['cleaned_tweet'].values.tolist())
train_tfidf_df = pd.DataFrame(train_tfidf.toarray(), columns=tfidf_vec.get_feature_names())
test_tfidf_df = pd.DataFrame(test_tfidf.toarray(), columns=tfidf_vec.get_feature_names())

logger.info("Dumping train and test TFIDF vectors for future modeling...")
pickle.dump( train_tfidf_df, open( data_dir+"train_tfidf_2.p", "wb" ) )
pickle.dump( test_tfidf_df, open( data_dir+"test_tfidf_2.p", "wb" ) )

del train_tfidf,test_tfidf,train_tfidf_df,test_tfidf_df
gc.collect()