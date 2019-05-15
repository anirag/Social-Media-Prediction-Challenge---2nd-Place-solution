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

def tweet_preprocesor(tweet):
    data = json.loads(tweet)
    req_cols = ["created_at","id","text","source","in_reply_to_status_id","in_reply_to_user_id","user",
           "coordinates","place","is_quote_status","retweeted_status","retweet_count","entities",
            "extended_entities","favorited","retweeted",
            "possibly_sensitive","lang"]

    data = data[req_cols]
    json_cols = ["user","coordinates","place","entities","extended_entities"]
def json_to_col(df):
    for col in json_cols:
        temp = df[col].swifter.apply(pd.Series)
        temp.columns = [col+'_'+str(c) for c in temp.columns]
        df = pd.concat([df, temp], axis=1).drop(col, axis=1)
    return df
