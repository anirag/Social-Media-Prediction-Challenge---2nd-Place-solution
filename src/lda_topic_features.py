import pandas as pd
import numpy as np
import gensim
from gensim import corpora, models, similarities
from nltk.tokenize import RegexpTokenizer
import pickle
from nltk.corpus   import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from textblob import TextBlob
import re,string
import config

import daiquiri,logging
daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger(__name__)

logger.info("loading data...")
data_dir = config.DATA_DIR
train = pickle.load( open( data_dir+"train_features.p", "rb" ) )
test = pickle.load( open( data_dir+"test_features.p", "rb" ) )
train = train[['id','text']]
test = test[['id','text']]


sw            = stopwords.words('english')
lemma         = WordNetLemmatizer()
def clean_text_and_tokenize(line):
    line   = re.sub(r'\$\w*', '', line)  # Remove tickers
    line   = re.sub(r'http?:.*$', '', line)
    line   = re.sub(r'https?:.*$', '', line)
    line   = re.sub(r'pic?.*\/\w*', '', line)
    line   = re.sub(r'^@\w*','', line)
    line   = re.sub(r'#([^\s]+)', r'\1', line)
    line   = re.sub(r'[' + string.punctuation + ']+', ' ', line)  # Remove puncutations like 's
    
    tokens = TweetTokenizer(strip_handles=True, reduce_len=True).tokenize(line)
    tokens = [w.lower() for w in tokens if w not in sw and len(w) > 2 and w.isalpha()]
    tokens = [lemma.lemmatize(word) for word in tokens]
    
    return " ".join(tokens)

logger.info("Preaparing Tweet text...")
import swifter
train['cleaned_tweet'] = train['text'].swifter.apply(lambda x: clean_text_and_tokenize(x))
test['cleaned_tweet'] = test['text'].swifter.apply(lambda x: clean_text_and_tokenize(x))

documents = train['cleaned_tweet'].values.tolist() + test['cleaned_tweet'].values.tolist()

#Inspired from
#https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

logger.info("Preparing for LDA model...")
tokenizer = RegexpTokenizer(r'\w+')
documents = [ tokenizer.tokenize(doc.lower()) for doc in documents ]
dictionary = corpora.Dictionary(documents)
dictionary.compactify()
# and save the dictionary for future use
dictionary.save(data_dir+'zindi.dict')
corpus = [dictionary.doc2bow(doc) for doc in documents]
corpora.MmCorpus.serialize(data_dir+'zindi_corpus_matrix.mm', corpus)

logger.info("Creating LDA model with 9 topics...")
mallet_path = data_dir+'mallet-2.0.8/bin/mallet'
corpus_filename = data_dir+'zindi_corpus_matrix.mm'
dict_filename = data_dir+'zindi.dict'
corpus = corpora.MmCorpus(corpus_filename)
dictionary = corpora.Dictionary.load(dict_filename)
num_topics = 9
model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=dictionary)

logger.info("Extracting dominant topic for each tweet using LDA model...")
def get_dominant_topic(ldamodel, corpus, texts, topic_number=0):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == topic_number:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                continue
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df.reset_index())


df_dominant_topic1 = get_dominant_topic(ldamodel=model, corpus=corpus, texts=documents, topic_number=0)
df_dominant_topic1.columns = ['Document_No', 'Dominant_Topic1', 'Topic_Perc_Contrib1', 'Keywords1', 'Text']
df_dominant_topic2 = get_dominant_topic(ldamodel=model, corpus=corpus, texts=documents, topic_number=1)
df_dominant_topic2.columns = ['Document_No', 'Dominant_Topic2', 'Topic_Perc_Contrib2', 'Keywords2', 'Text']
df_dominant_topic3 = get_dominant_topic(ldamodel=model, corpus=corpus, texts=documents, topic_number=2)
df_dominant_topic3.columns = ['Document_No', 'Dominant_Topic3', 'Topic_Perc_Contrib3', 'Keywords3', 'Text']

topic_features = df_dominant_topic1[['Document_No','Dominant_Topic1','Topic_Perc_Contrib1']].merge(
df_dominant_topic2[['Document_No','Dominant_Topic2','Topic_Perc_Contrib2']], on='Document_No', how='left')
topic_features = topic_features.merge(df_dominant_topic3[['Document_No','Dominant_Topic3','Topic_Perc_Contrib3']], on='Document_No', how='left')
topic_features.drop('Document_No',axis=1,inplace=True)

train_topics = topic_features.iloc[:train.shape[0]]
test_topics = topic_features.iloc[train.shape[0]:]

logger.info("Dumping train and test topic features...")
train_topics.to_csv(data_dir+'train_topics_features.csv',index=False)
test_topics.to_csv(data_dir+'test_topics_features.csv',index=False)