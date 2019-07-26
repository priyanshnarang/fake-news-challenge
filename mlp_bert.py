#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 03:06:32 2019

@author: b67singh
"""

import os
import keras
from keras import callbacks
import tensorflow as tf
from keras.layers import Dense, Input, Dropout
from keras.models import load_model, Model
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical


def build_model():
    input_layer = Input(shape=(804,), dtype='float32', name='input_layer')
    
    dense1 = Dense(128, activation = 'relu', name = 'dense_1')(input_layer) 
    drop1 = Dropout(0.4, name = 'droput_1')(dense1)
    
    dense2 = Dense(64, activation = 'relu', name = 'dense_2')(drop1) 
    drop2 = Dropout(0.3, name = 'dropout_2')(dense2)
    
    output_layer_2 = Dense(2, activation='softmax', name = 'output_layer_2')(drop2)
    output_layer_4 = Dense(4, activation='softmax', name = 'output_layer_4')(drop2)
    
    model = Model(inputs = input_layer, outputs = [output_layer_2, output_layer_4])
    
    adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
    
    model.compile(optimizer=adam,
                  loss={'output_layer_2': 'categorical_crossentropy', 'output_layer_4': 'categorical_crossentropy'},
                  loss_weights={'output_layer_2': 0.25, 'output_layer_4': 0.75},
                  metrics = {'output_layer_2': 'accuracy', 'output_layer_4': 'accuracy'})
    
    return model




def MLP_bert_training(X, y):
    y_4_categorical = to_categorical(y,4)
    
    y_train_2 = [0 if ele  == '3' else 1 for ele in y]     
    y_2_categorical = to_categorical(y_train_2)

    if not os.path.isfile('models/model_mlp.h5'):   
        
        callback = [EarlyStopping(monitor='val_loss', verbose=1, patience = 3),ModelCheckpoint('model_mlp.h5', monitor='val_loss', verbose=1, save_best_only=True)]

  
        model_mlp = build_model()    
        custom_weights_2 ={0: 1, 1: 5.72175}
        custom_weights_4 ={0:12.93617, 1:43.50595, 2:11.1020, 3:1}
        
        model_mlp.fit(X,
          {'output_layer_2': y_2_categorical, 'output_layer_4': y_4_categorical},verbose=2,
          epochs=100, batch_size=128, validation_split = .2, callbacks = callback, class_weight=[custom_weights_2, custom_weights_4])
    
        model_mlp.save('models/model_mlp.h5')    
        
        return model_mlp
    
    model_mlp = load_model('models/model_mlp.h5')
        
    return model_mlp


def MLP_bert_predict(X, clf_mlp):
    
    y_pred_2, y_pred_4 = clf_mlp.predict(X)
    return y_pred_4
    