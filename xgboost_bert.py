#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 02:04:02 2019

@author: Priyansh Narang
"""
import os
import time
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from numpy import inf
import pickle
import numpy as np

class XGBoostFNC:
    def __init__(self, X_train, y_train, X_test):
        self.x_train = X_train
        self.y_train = y_train
        self.x_test = X_test
        self.y_pred = None
        self.model = None
        if not os.path.isfile('models/model_xgboost.pickle'):
            self.model_exists = False
        else:
            self.model_exists = True
        
        print("XG Boost model: ", self.model_exists)
    
    def persist_model(self):
        try:
            with open('models/model_xgboost.pickle ', 'wb') as handle:
                pickle.dump(self.model, handle)
                
        except Exception as e:
            print(e)
    
    def load_model(self):
        if self.model_exists == False:
            self.train_model()
            
        elif self.model_exists:
            with open('models/model_xgboost.pickle', 'rb') as handle:
                model_xgboost = pickle.load(handle)
                self.model = model_xgboost
            
        return True
    
    def train_model(self):
        try:
            # Load WMD Features for Train Set
            wmd_train = np.load('features/wmd_train.npy')
            wmd_train[wmd_train == inf] = np.nanmax(wmd_train[wmd_train != np.inf])
            X_train_wmd = np.c_[self.x_train, wmd_train.reshape(-1,1)]
            
            start_time = time.time()
            print("Training XG-Boost Model on X: {0} and Y: {1}".format(X_train_wmd.shape, len(self.y_train)))
            
            # Train a XG-Boost Classifier
            model_xgboost = XGBClassifier(n_estimators=1000, learning_rate=0.1, n_jobs = -1)     
            model_xgboost.fit(X_train_wmd, self.y_train)
            
            print("Model finished training in {0} seconds".format(time.time()-start_time))
            
            # Change class properties to reflect changes
            self.model_exists = True
            self.model = model_xgboost
            self.persist_model()
        
        except Exception as e:
            print(e)
    
    def predict(self):
        try:
            if self.load_model():
            
                # Load WMD features for Test Set
                wmd_test = np.load('features/wmd_test.npy')
                wmd_test[wmd_test == inf] = np.nanmax(wmd_test[wmd_test != np.inf])
                X_test_wmd = np.c_[self.x_test, wmd_test.reshape(-1,1)]
                
                # Predict y values
                y_pred = self.model.predict_proba(X_test_wmd)
                self.y_pred = y_pred
                return y_pred
        
        except Exception as e:
            print(e)
    
