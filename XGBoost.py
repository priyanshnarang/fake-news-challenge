#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 02:04:02 2019

@author: b67singh
"""
import os
from xgboost import XGBClassifier
from numpy import inf
import pickle
import numpy as np



def XGBoost_training(X, y):
    
    if not os.path.isfile('models/model_xgboost.pickle'):   
        train_wmd = np.load('features/wmd_train.npy')
        wmd_train[wmd_train == inf] = np.nanmax(wmd_train[wmd_train != np.inf])
        X_train_wmd = np.c_[X, wmd_train.reshape(-1,1)]
        
        model_xgboost = XGBClassifier(n_estimators=1000, learning_rate=0.1, n_jobs = -1)     
        model_xgboost.fit(X_train_wmd, y)
        
        with open('models/model_xgboost.pickle ', 'wb') as handle:
            pickle.dump(model_xgboost, handle)
    
    with open('models/model_xgboost.pickle', 'rb') as handle:
        model_xgboost = pickle.load(handle)
        
    return model_xgboost


def XGBoost(X, clf_xgb):
    
    wmd_test = np.load('features/wmd_test.npy')
    wmd_test[wmd_test == inf] = np.nanmax(wmd_test[wmd_test != np.inf])
    X_test_wmd = np.c_[X, wmd_test.reshape(-1,1)]
    y_pred = clf_xgb.predict(X_test_wmd)
    return y_pred
    