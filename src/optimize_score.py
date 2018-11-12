import matplotlib
matplotlib.use('Agg')
import ROOT
import os, sys, errno
import math
from array import array
import pandas
import numpy
import uproot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Dropout, Concatenate, Lambda
import matplotlib.pyplot as plt
from keras import backend as K
from tensorflow import where, greater, abs, zeros_like, exp
import tensorflow as tf
from keras.losses import kullback_leibler_divergence, categorical_crossentropy
ROOT.gStyle.SetOptStat(0)


class Optimizer(object):

    def __init__(self, path):
        self.path = path
        self.var_list = ['mass', 'DY_prediction', 'ttbar_prediction', 'ggH_prediction', 'VBF_prediction', 'weight']
        self.training_labels = ['DY_prediction', 'ttbar_prediction', 'ggH_prediction', 'VBF_prediction']
        self.true_labels = ['DY', 'ttbar', 'ggH', 'VBF']
        self.sig_mask = [0.0,0.0,1.0,1.0]
        self.bkg_mask = [1.0,1.0,0.0,0.0]
        self.sig_labels = ['ggH', 'VBF']
        self.bkg_labels = ['DY', 'ttbar']
        self.df = pandas.DataFrame()
        self.threshold = 0

    def convert_to_df(self):
        with uproot.open(self.path) as f: 

            trees = {
                'DY':       f["tree_ZJets_MG"],
                'ttbar':    f["tree_tt_ll_AMC"],
                'ggH':      f["tree_H2Mu_gg"],
                'VBF':      f["tree_H2Mu_VBF"]

            }

            for label, tree in trees.iteritems():
                df_for_tree = pandas.DataFrame()
                for var in self.var_list:
                    up_var = tree[var].array()
                    df_for_tree[var] = up_var

                for true_label in self.true_labels:
                    if  true_label==label:
                        df_for_tree[true_label] = df_for_tree['weight']
                    else:
                        df_for_tree[true_label] = 0

                self.df = pandas.concat([self.df, df_for_tree])
        self.df = shuffle(self.df)
        print self.df


    def train(self):
        inputs = Input(shape=(4,), name = 'input') 
        x = Dense(50, name = 'layer_1', activation='relu')(inputs)
        x = Dense(50, name = 'layer_2', activation='relu')(inputs)
        outputs = Dense(1, name = 'output')(x)

        lambdaLayer = Lambda(lambda x: 0*x, name='lambda')(inputs)
        def slicer(x):
            return x[:,0:3]    
        lambdaLayer = Lambda(slicer)(lambdaLayer)

        final_outputs = Concatenate()([outputs, lambdaLayer]) # order is important


        model = Model(inputs=inputs, outputs=final_outputs)
        model.compile(loss=[self.loss],                                  
                      optimizer='adam'
                      , metrics=[self.sig_above, self.bkg_above])   
 
        model.summary()
        history = model.fit(            
                                    self.df[self.training_labels].values,
                                    self.df[self.true_labels].values,
                                    epochs=300, 
                                    batch_size=2048, 
                                    verbose=1,
                                    validation_split=0.25,
                                    shuffle=True)


    def score(self, y_true_, y_pred):
        max_score = K.max(y_pred[:,0])
        min_score = K.min(y_pred[:,0])
        return min_score



    def sig_above(self, y_true, y_pred):
        x = y_pred[:, 0] # other 3 bins are dummy
        x = (tf.sign(x - self.threshold)+1) / 2                                             
        y = K.dot(tf.diag(x), y_true)         
        sig_sum = tf.reduce_sum(K.dot(y, tf.diag(self.sig_mask)))
        return sig_sum

    def bkg_above(self, y_true, y_pred):
        x = y_pred[:, 0] # other 3 bins are dummy
        x = (tf.sign(x - self.threshold)+1) / 2                                             
        y = K.dot(tf.diag(x), y_true)         
        bkg_sum = tf.reduce_sum(K.dot(y, tf.diag(self.bkg_mask)))
        return bkg_sum


    def loss(self, y_true, y_pred):     
        x = y_pred[:, 0] # other 3 bins are dummy
        x = (tf.sign(x - self.threshold)+1) / 2         # cut on output to take into account only some high-score events
                                                        # dimension of x: (batch_size, 1)
        y = K.dot(tf.diag(x), y_true)                   # leave only those y for which dnn score is above threshold
                                                        # dimension of y: (batch_size, 4)
        bkg_sum = tf.reduce_sum(K.dot(y, tf.diag(self.bkg_mask)))
        sig_sum = tf.reduce_sum(K.dot(y, tf.diag(self.sig_mask)))
        return bkg_sum/sig_sum



input_path = "output/Run_2018-11-05_13-52-32/Keras_multi/model_50_D2_25_D2_25_D2/root/output_test.root"
o = Optimizer(input_path)
o.convert_to_df()
o.train()