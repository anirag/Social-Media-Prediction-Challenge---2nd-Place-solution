Repository for 2nd place solution for Social Media Prediction Challenge hosted on Zindi.Africa

To get submission that scored 2nd on private and 2nd on public leaderboards, Add "test_questions.json" and "train.json" to the data folder then make apporpriate changes to "config.py" and then running `bash run_all.sh` will save the submission file in the results folder

Note:

Final Model consists of four sets of features: hand-crafted, tfidf-word vector of tweet text, tfidf-char vector of tweet-text and topic modelling features.

hand_crafted (selected_features2)+ tfidf-word + topic features should produce a result of ~3.268
hand_crafted (selected_features2)+ tfidf-word + tfidf-char + topic features should produce a result of ~3.266
hand_crafted (selected_features1) + topic features should produce a result of ~3.272

To modify what feeatures to include, use config.py 

