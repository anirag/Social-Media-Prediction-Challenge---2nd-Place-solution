import pandas as pd
import numpy as np
from nltk.corpus   import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from textblob import TextBlob
import re,string
from bs4 import BeautifulSoup
import json
import swifter
import pickle
import config

import daiquiri,logging
daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger(__name__)

data_dir = config.DATA_DIR
trainpath = data_dir+"train.json"
testpath = data_dir+"test_questions.json"
print(data_dir)
logger.info("Reading in train and test json files...")
train = pd.read_json(trainpath, orient='columns')
test = pd.read_json(testpath, orient='columns')

req_cols = ["created_at","id","text","source","in_reply_to_status_id","in_reply_to_user_id","user",
           "coordinates","place","is_quote_status","retweeted_status","retweet_count","entities",
            "extended_entities","favorited","retweeted",
            "possibly_sensitive","lang"]

train = train[req_cols]
test = test[[c for c in req_cols if c in test.columns]]

logger.info("Converting json fields to columns...")
json_cols = ["user","coordinates","place","entities","extended_entities"]
def json_to_col(df):
    for col in json_cols:
        temp = df[col].swifter.apply(pd.Series)
        temp.columns = [col+'_'+str(c) for c in temp.columns]
        df = pd.concat([df, temp], axis=1).drop(col, axis=1)
    return df

train = json_to_col(train)
test = json_to_col(test)

for row in train.loc[train.extended_entities_media.isnull(), 'extended_entities_media'].index:
    train.at[row, 'extended_entities_media'] = []
for row in test.loc[test.extended_entities_media.isnull(), 'extended_entities_media'].index:
    test.at[row, 'extended_entities_media'] = []

logger.info("Defining few utility functions ...")

def get_source(x):
    soup = BeautifulSoup(x,"lxml")
    for link in soup.find_all('a'):
        return(link.text)

def sentiment_analysis_basic(tweet):
    analysis  = TextBlob(tweet)
    sentiment = analysis.sentiment.polarity
    
    return(sentiment)

def subjectivity_analysis_basic(tweet):
    analysis  = TextBlob(tweet)
    subjectivity = analysis.sentiment.subjectivity
    
    return(subjectivity)

def informativeness(x):
    if 'linktoken' in x:
        return 1
    else:
        return(len(x)/280.)


regexp = {"RT": "^RT", "MT": r"^MT", "ALNUM": r'(?<![@\w])@(\w{1,25})',
              "HASHTAG": r"(#[\w\d]+)", "URL": r"([https://|http://]?[a-zA-Z\d\/]+[\.]+[a-zA-Z\d\/\.]+)",
              "SPACES": r"\s+"}

regexp = dict((key, re.compile(value)) for key, value in regexp.items())
def getAttributeRT(tweet):
        """ see if tweet is a RT """
        return re.search(regexp["RT"], tweet.strip()) is not None


def getAttributeMT(tweet):
    """ see if tweet is a MT """
    return re.search(regexp["MT"], tweet.strip()) is not None


def getUserHandles(tweet):
    """ given a tweet we try and extract all user handles in order of occurrence"""
    return re.findall(regexp["ALNUM"], tweet)

    
def getHashtags(tweet):
    """ return all hashtags"""
    return re.findall(regexp["HASHTAG"], tweet)


def getURLs(tweet):
    r""" URL : [http://]?[\w\.?/]+"""
    listed = re.findall(regexp["URL"], tweet)
    c=0
    for e in listed:
        if "..." in e:
            continue
        c+=1
    return(c)

def tweet_features(df):
    df['number_of_hastags'] = df['entities_hashtags'].swifter.apply(lambda x: len(x))
    df['number_of_symbols'] = df['entities_symbols'].swifter.apply(lambda x: len(x))
    df['number_of_urls'] = df['entities_urls'].swifter.apply(lambda x: len(x))
    df['number_of_mentions'] = df['entities_user_mentions'].swifter.apply(lambda x: len(x))
    df['number_of_medias'] = df['extended_entities_media'].swifter.apply(lambda x: len(x))
    df['length_of_the_tweet'] = df['text'].swifter.apply(lambda x: len(x))
    df['number_of_words'] = df['text'].swifter.apply(lambda x: len(x.split()))
    df['is_tweet_a_reply'] = df['in_reply_to_status_id'].swifter.apply(lambda x: 0 if np.isnan(x) else 1)
    df['is_tweet_a_quote'] = df['is_quote_status'].astype(int)
    df['is_tweet_sensitive'] = df['possibly_sensitive'].swifter.apply(lambda x: -1 if np.isnan(x) else x)
    df['tweet_sentiment'] = df['text'].swifter.apply(lambda x: sentiment_analysis_basic(x))
    df['tweet_subjectivity'] = df['text'].swifter.apply(lambda x: subjectivity_analysis_basic(x))
    df['tweet_informativeness'] = df['text'].swifter.apply(lambda x: informativeness(x))
    df['tweet_location'] = df['user_location']
    df['tweet_time'] = pd.to_datetime(df['created_at'])
    df['user_time'] = pd.to_datetime(df['user_created_at'])
    df['tweet_created_year'] = df['tweet_time'].dt.year
    df['tweet_created_hour'] = df['tweet_time'].dt.hour
    df['tweet_created_month'] = df['tweet_time'].dt.month
    df['tweet_created_day'] = df['tweet_time'].dt.weekday
    df['sin_hr'] = np.sin(2*np.pi*df.tweet_created_hour/24)
    df['cos_hr'] = np.cos(2*np.pi*df.tweet_created_hour/24)
    df['sin_mth'] = np.sin(2*np.pi*df.tweet_created_month/12)
    df['cos_mth'] = np.cos(2*np.pi*df.tweet_created_month/12)
    df['sin_wday'] = np.sin(2*np.pi*df.tweet_created_day/7)
    df['cos_wday'] = np.cos(2*np.pi*df.tweet_created_day/7)
    
    df['tweet_time'] = df['tweet_time'].dt.tz_convert(None)
    df['user_time'] = df['user_time'].dt.tz_convert(None)
    df['days_after_creation'] = (df['tweet_time'] - df['user_time']).astype('timedelta64[D]')
    df['user_activity'] = df['user_statuses_count']/(1.+ df['days_after_creation'])
    df['user_reliability'] = df['user_friends_count']/(df['user_friends_count']+df['user_followers_count'])
    
    df.drop(['created_at','tweet_time','user_location','user_time',
             'tweet_created_hour','tweet_created_month','tweet_created_day'], axis=1, inplace=True)
    df['tweet_source'] = df['source'].swifter.apply(lambda x: get_source(x))
    #additional
    df['is_RT'] = df['text'].swifter.apply(lambda x: getAttributeRT(x))
    df['is_MT'] = df['text'].swifter.apply(lambda x: getAttributeMT(x))
    df['num_UH_from_text'] = df['text'].swifter.apply(lambda x: len(getUserHandles(x)))
    df['num_HT_from_text'] = df['text'].swifter.apply(lambda x: len(getHashtags(x)))
    df['num_URLS_from_text'] = df['text'].swifter.apply(lambda x: getURLs(x))
    df['is_question'] = df['text'].swifter.apply(lambda x: 1 if '?' in x else 0)

    return df

logger.info("Creating user,tweet and time features...")

train = tweet_features(train)
test = tweet_features(test)

train.tweet_location.replace(to_replace=['Lagos, Nigeria.',
                                 'Everywhere you go.','Everywhere you go ','KENYA'],
                     value=['Lagos, Nigeria','Everywhere you go','Everywhere you go','Kenya'], inplace=True)
test.tweet_location.replace(to_replace=['Lagos, Nigeria.',
                                 'Everywhere you go.','Everywhere you go ','KENYA'],
                     value=['Lagos, Nigeria','Everywhere you go','Everywhere you go','Kenya'], inplace=True)


social_features = ['user_name','user_followers_count','user_friends_count', 'user_listed_count',
                   'user_favourites_count','user_statuses_count','user_verified',
                  'user_following','days_after_creation','user_activity','user_reliability']
tweet_features = ['number_of_hastags','number_of_mentions','number_of_urls','length_of_the_tweet','number_of_medias','number_of_symbols',
                  'tweet_source','is_tweet_a_reply','is_tweet_a_quote','is_tweet_sensitive','number_of_words',
                  'tweet_location','tweet_subjectivity','tweet_informativeness','tweet_sentiment',
                  'sin_hr','sin_mth','sin_wday','cos_hr','cos_mth','cos_wday','tweet_created_year']
text_features = ['is_RT','is_MT','num_UH_from_text','num_HT_from_text','num_URLS_from_text','is_question']
other_features = ['id','text','retweet_count']

features = other_features+social_features+tweet_features+text_features

train = train[features]
test = test[[c for c in features if c in test.columns]]

logger.info("Cleaning tweet text...")

sw            = stopwords.words('english')
lemma         = WordNetLemmatizer()

def clean_text_and_tokenize(line):
    line   = re.sub(r'\$\w*', '', line)  # Remove tickers
    line   = re.sub(r'http?:.*$', 'linktoken', line)
    line   = re.sub(r'https?:.*$', 'linktoken', line)
    line   = re.sub(r'pic?.*\/\w*', 'pictoken', line)
    line   = re.sub(r'^@\w*','mentiontoken', line)
    line   = re.sub(r'#([^\s]+)', r'\1', line)
    line   = re.sub(r'[' + string.punctuation + ']+', ' ', line)  # Remove puncutations like 's
    
    tokens = TweetTokenizer(strip_handles=True, reduce_len=True).tokenize(line)
    tokens = [w.lower() for w in tokens if w not in sw and len(w) > 2 and w.isalpha()]
    tokens = [lemma.lemmatize(word) for word in tokens]
    
    return " ".join(tokens)

train['cleaned_tweet'] = train['text'].swifter.apply(lambda x: clean_text_and_tokenize(x))
test['cleaned_tweet'] = test['text'].swifter.apply(lambda x: clean_text_and_tokenize(x))

logger.info("Adding valence,arousal,dominance features based on cleaned tweet...")

vad = pd.read_csv(data_dir+'BRM-emot-submit.csv')
afinn = pd.read_csv(data_dir+'AFINN-111.txt',sep="\t",header=None)
afinn.columns = ['word','afinn']

afinn_dict = dict(zip(afinn.word,afinn.afinn))
valence_dict = dict(zip(vad.Word,vad['V.Mean.Sum']))
dominance_dict = dict(zip(vad.Word,vad['D.Mean.Sum']))
arousal_dict = dict(zip(vad.Word,vad['A.Mean.Sum']))
train['valence'] = train['cleaned_tweet'].apply(lambda x: np.mean([valence_dict.get(w,0) for w in x.split()]))
test['valence'] = test['cleaned_tweet'].apply(lambda x: np.mean([valence_dict.get(w,0) for w in x.split()]))
train['arousal'] = train['cleaned_tweet'].apply(lambda x: np.mean([arousal_dict.get(w,0) for w in x.split()]))
test['arousal'] = test['cleaned_tweet'].apply(lambda x: np.mean([arousal_dict.get(w,0) for w in x.split()]))
train['dominance'] = train['cleaned_tweet'].apply(lambda x: np.mean([dominance_dict.get(w,0) for w in x.split()]))
test['dominance'] = test['cleaned_tweet'].apply(lambda x: np.mean([dominance_dict.get(w,0) for w in x.split()]))
train['afinn_score'] = train['cleaned_tweet'].apply(lambda x: sum([afinn_dict.get(w,0) for w in x.split()]))
test['afinn_score'] = test['cleaned_tweet'].apply(lambda x: sum([afinn_dict.get(w,0) for w in x.split()]))


logger.info("Dumping the processed train and test dataframes to pickle file...")

pickle.dump( train, open( data_dir+"train_features.p", "wb" ) )
pickle.dump( test, open( data_dir+"test_features.p", "wb" ) )

