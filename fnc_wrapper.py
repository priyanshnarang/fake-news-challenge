# -*- coding: utf-8 -*-
import numpy as np
import warnings
from utils.dataset import DataSet
from utils.score import report_score, LABELS, score_submission
from utils.system import parse_params, check_version
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features
from feature_engineering import sentiment_features, ner_features, bert_features, cosine_features
from xgboost_bert import XGBoostFNC
from mlp_bert import MLP_bert_training, MLP_bert_predict

warnings.simplefilter("ignore")

def generate_features(stances,dataset,name):
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])
    
    
    X_sentiment = gen_or_load_feats(sentiment_features, h, b, "features/sentiment."+name+".npy")
    X_ner = gen_or_load_feats(ner_features, h, b, "features/ner."+name+".npy")
    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")
    X_bert = bert_features("features/combined_bert_"+name+".csv")
    X_cosine = cosine_features(name, "features/cosine."+name+".npy")
    
    X = np.c_[X_refuting, X_overlap, X_hand, X_sentiment, X_ner, X_polarity, X_bert, X_cosine]
    
    # X_train_79 = pd.concat([refuting_features, overlap_features, hand_features.loc[:,0:3],hand_features.loc[:,16:], 
    #sentiment_features, ner_features, polarity_features, train_combined, pd.Series(cosine_sim_train)], axis=1)
    
    return X,y

if __name__ == "__main__":
    check_version()
    parse_params()

    #Load the training dataset and generate folds
    d = DataSet()
    X_train, y_train = generate_features(d.stances, d, "train")
    
    # Load competition dataset
    competition_dataset = DataSet("competition_test")
    X_competition, y_competition = generate_features(competition_dataset.stances, competition_dataset, "competition")
    
    # Train XG-Boost classifier
    xgb_clf = XGBoostFNC(np.c_[X_train[:,0:25], X_train[:,30:]], y_train, np.c_[X_competition[:,0:25], X_competition[:,30:]])
    #xgb_clf = XGBoostFNC(X_train, y_train, X_competition)
    y_pred_xgb = xgb_clf.predict()
    
    # Train MLP Classifier
    mlp_clf = MLP_bert_training(X_train, y_train)
    y_pred_mlp = MLP_bert_predict(X_competition, mlp_clf)
    
    # Average the results
    predicted_probs = 0.45 * y_pred_xgb + 0.55 * y_pred_mlp
    predicted_labels = [np.argmax(x) for x in predicted_probs]
    
    predicted = [LABELS[int(a)] for a in predicted_labels]
    actual = [LABELS[int(a)] for a in y_competition]
    
    print("Scores on the test set")
    report_score(actual,predicted)
    


    
    



