import pandas as pd
import numpy as np
import pickle
import config

import daiquiri,logging
daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger(__name__)

logger.info("Reading in train and test dataframes...")
data_dir = config.DATA_DIR
results_dir = config.RESULTS_DIR
train_features = pickle.load( open( data_dir+"train_features.p", "rb" ) )
test_features = pickle.load( open( data_dir+"test_features.p", "rb" ) )

logger.info("Preparing Hand crafted features...")
train_ids = train_features.id.values
train_y = train_features.retweet_count.values

test_ids = test_features.id.values

train_features.drop(['id','retweet_count','text','cleaned_tweet'], axis=1,inplace=True)
test_features.drop(['id','text','cleaned_tweet'], axis=1, inplace=True)

for col in ['user_verified','user_following']:
    train_features[col] = 1*train_features[col] 
    test_features[col] = 1*test_features[col] 

categorical_cols = [col for col in train_features.columns if train_features[col].dtype=='object']
from sklearn import preprocessing
for col in categorical_cols:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_features[col].values.astype('str')) + list(test_features[col].values.astype('str')))
    train_features[col] = lbl.transform(list(train_features[col].values.astype('str')))
    test_features[col] = lbl.transform(list(test_features[col].values.astype('str')))

include_tfidf = config.INCLUDE_TFIDF
include_tfidf2 = config.INCLUDE_TFIDF_CHAR
include_w2v = config.INCLUDE_W2V
include_topic_feature = config.INCLUDE_TOPIC

if include_tfidf:
  logger.info("Preparing TFIDF word model...")
  train_tfidf = pickle.load( open( data_dir+"train_tfidf.p", "rb" ) )
  test_tfidf = pickle.load( open( data_dir+"test_tfidf.p", "rb" ) )
  train_tfidf.columns = train_tfidf.columns.str.replace(' ', '_')
  test_tfidf.columns = test_tfidf.columns.str.replace(' ', '_')
  train_tfidf.columns = ["tfidf_1_"+c for c in train_tfidf.columns]
  test_tfidf.columns = ["tfidf_1_"+c for c in test_tfidf.columns]

if include_tfidf2:
  logger.info("Preparing TFIDF char model...")
  train_tfidf2 = pickle.load( open( data_dir+"train_tfidf_2.p", "rb" ) )
  test_tfidf2 = pickle.load( open( data_dir+"test_tfidf_2.p", "rb" ) )
  train_tfidf2.columns = train_tfidf2.columns.str.replace(' ', '_')
  test_tfidf2.columns = test_tfidf2.columns.str.replace(' ', '_')
  train_tfidf2.columns = ["tfidf_2_"+c for c in train_tfidf2.columns]
  test_tfidf2.columns = ["tfidf_2_"+c for c in test_tfidf2.columns]

if include_topic_feature:
  logger.info("Preparing topic model features...")
  train_topics_features = pd.read_csv(data_dir+"train_topics_features.csv")
  test_topics_features = pd.read_csv(data_dir+"test_topics_features.csv")
  sel = ['Dominant_Topic1', 'Topic_Perc_Contrib1','Dominant_Topic2', 'Topic_Perc_Contrib2',
  'Dominant_Topic3', 'Topic_Perc_Contrib3' ]

if include_w2v:
  logger.info("Preparing w2v model features...")
  train_word2vec = pickle.load( open( data_dir+"train_word2vec.p", "rb" ) )
  test_word2vec = pickle.load( open( data_dir+"test_word2vec.p", "rb" ) )

selected_features1 = config.selected_features1
selected_features2 = config.selected_features2

logger.info("Adding all the features...")
#train = train_features[selected_features1]
#test = test_features[selected_features1]
train = train_features[selected_features2]
test = test_features[selected_features2]

if include_tfidf:
    train = pd.concat([train,train_tfidf], axis=1)
    test = pd.concat([test,test_tfidf], axis=1)
if include_tfidf2:
    train = pd.concat([train,train_tfidf2], axis=1)
    test = pd.concat([test,test_tfidf2], axis=1)
if include_w2v:
    train = pd.concat([train,train_word2vec], axis=1)
    test = pd.concat([test,test_word2vec], axis=1)
if include_topic_feature:
    train = pd.concat([train,train_topics_features[sel]], axis=1)
    test = pd.concat([test,test_topics_features[sel]], axis=1)

print(f"Final shape of train and test dataframe: {train.shape},{test.shape}")

logger.info("Setting the parameters for the LightGBM model...")
learning_rate = 0.1
num_leaves = 15
#min_data_in_leaf = 2000
feature_fraction = 0.8
num_boost_round = 10000
params = {"objective": "regression_l1",
          "boosting_type": "gbdt",
          "learning_rate": learning_rate,
          "num_leaves": num_leaves,
          "feature_fraction": feature_fraction,
          "verbosity": 1,
          #"drop_rate": 0.1,
          #"max_drop": 50,
          #"min_child_samples": 10,
          #"min_child_weight": 150,
          #"min_split_gain": 0,
          "subsample": 0.8,
          
          }

cv_only = True
save_cv = True
full_train = False

from sklearn import metrics
import lightgbm as lgbm
from sklearn.model_selection import StratifiedKFold,KFold
NFOLDS = 5
NROUNDS = 2
kfold = KFold(n_splits=NFOLDS, shuffle=True, random_state=2018)


logger.info("Modelling Starts...")
x_score = []
final_cv_train = np.zeros(len(train_y))
final_cv_pred = np.zeros(len(test_ids))
for s in range(NROUNDS):
    cv_train = np.zeros(len(train_y))
    cv_pred = np.zeros(len(test_ids))

    params['seed'] = s

    if cv_only:
        kf = kfold.split(train, train_y)

        best_trees = []
        fold_scores = []

        for i, (train_fold, validate) in enumerate(kf):
            print("Running... Seed",s," Fold: ",i+1)
            X_train, X_validate, label_train, label_validate = \
                train.iloc[train_fold, :], train.iloc[validate, :], train_y[train_fold], train_y[validate]
            dtrain = lgbm.Dataset(X_train, label_train)
            dvalid = lgbm.Dataset(X_validate, label_validate, reference=dtrain)
            bst = lgbm.train(params, dtrain, num_boost_round, valid_sets=[dtrain,dvalid], verbose_eval=100,early_stopping_rounds=50)
            bst.model_to_string(f'lightgbm_model_{s}_{i}.p')
            best_trees.append(bst.best_iteration)
            cv_pred += bst.predict(test, num_iteration=bst.best_iteration)
            cv_train[validate] += bst.predict(X_validate)

            score = metrics.mean_absolute_error(label_validate, cv_train[validate])
            print("validation MAE: ",score)
            fold_scores.append(score)

        cv_pred /= NFOLDS
        final_cv_train += cv_train
        final_cv_pred += cv_pred

        print("cv score:")
        print(metrics.mean_absolute_error(train_y, cv_train))
        print("current score:", metrics.mean_absolute_error(train_y, final_cv_train / (s + 1.)), s+1)
        print(fold_scores)
        print(best_trees, np.mean(best_trees))

        x_score.append(metrics.mean_absolute_error(train_y, cv_train))
print(f"Scores: {x_score}")
print(f"MAE after running for {len(x_score)} trials, each for {NFOLDS} folds:  mean MAE {np.mean(x_score)}, std MAE: {np.std(x_score)}")

logger.info("Submission and CV predictions...")
sub_df = pd.DataFrame()
sub_df['id'] = test_ids
sub_df['retweet_count'] = final_cv_pred / NROUNDS
sub_df.loc[sub_df['retweet_count']<0,'retweet_count']=0
pd.DataFrame({'id': test_ids, 'retweet_count': final_cv_pred / NROUNDS}).to_csv(results_dir+'final_submission.csv', index=False)
#pd.DataFrame({'id': train_ids, 'retweet_count': final_cv_train / 2.}).to_csv(results_dir+'lgbm_cv_avg_all_features.csv', index=False)

