# Social Media Prediction Challenge

The objective of this competition is to create a model to predict the number of retweets a tweet will get on Twitter. The data used to train the model will be approximately 2,400 tweets each from 38 major banks and mobile network operators across Africa.

A machine learning model to predict retweets would be valuable to any business that uses social media to share important information and messages to the public. This model can be used as a tool to help businesses better tailor their tweets to ensure maximum impact and outreach to clients and non-clients.

## Getting Started

These instructions will get you a Machine Learning model that predicts the number of retweets given a tweet json

### Prerequisites

Install the python packages in reuirements.txt file. Add "test_questions.json" and "train.json" to the data folder then make apporpriate changes to "config.py" and then running `bash run_all.sh` will save the submission file in the results folder

### Note

Final Model consists of four sets of features: hand-crafted, tfidf-word vector of tweet text, tfidf-char vector of tweet-text and topic modelling features.

* hand_crafted (selected_features2)+ tfidf-word + topic features should produce a result of ~3.268
* hand_crafted (selected_features2)+ tfidf-word + tfidf-char + topic features should produce a result of ~3.266
* hand_crafted (selected_features1) + topic features should produce a result of ~3.272

To modify what features to include, use config.py 
