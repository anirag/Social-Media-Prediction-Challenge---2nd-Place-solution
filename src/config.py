# Please change this appropriately
DATA_DIR = "../solution/data/"
RESULTS_DIR = "../solution/results/"

INCLUDE_TFIDF = True
INCLUDE_TFIDF_CHAR = True
INCLUDE_TOPIC = True
INCLUDE_W2V = False

selected_features1 = ['user_followers_count', 'user_friends_count', 'user_listed_count', 'user_favourites_count', 'user_statuses_count', 'user_verified',
       'number_of_hastags', 'number_of_mentions', 'number_of_urls', 'number_of_symbols','number_of_medias','length_of_the_tweet', 'tweet_source', 
       'is_tweet_a_reply','is_tweet_a_quote', 'is_tweet_sensitive', 'number_of_words','tweet_location', 'sin_hr', 'sin_mth', 'sin_wday', 'cos_hr', 
       'cos_mth', 'cos_wday', 'tweet_sentiment', 'tweet_subjectivity','user_following','days_after_creation','user_activity','user_reliability',
       'valence','arousal','dominance','afinn_score']

selected_features2 = ['user_followers_count', 'user_friends_count', 'user_listed_count', 'user_favourites_count', 'user_statuses_count', 'user_verified',
       'number_of_hastags', 'number_of_mentions', 'number_of_urls', 'length_of_the_tweet', 'tweet_source', 'user_following',
       'is_tweet_a_reply','is_tweet_a_quote', 'is_tweet_sensitive', 'number_of_words','tweet_location', 'sin_hr', 'sin_mth', 'sin_wday', 'cos_hr', 
       'cos_mth', 'cos_wday', 'tweet_sentiment']