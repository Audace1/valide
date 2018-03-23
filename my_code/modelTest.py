#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 08:04:23 2017

@author: isabelleguyon

This is an example of program that tests the Iris challenge Classifier class.
Another style is to incorporate the test as a main function in the Data manager class itself,
but having both is not incompatible (a smaller test with the program and a more developped use case here).

This module could perform hyper-parameter selection by cross-validation
See: http://scikit-learn.org/stable/modules/cross_validation.html.
"""

from sys import path
path.append ("../scoring_program")    # Contains libraries you will need
path.append ("../ingestion_program")  # Contains libraries you will need

from data_manager import DataManager
from data_converter import convert_to_num 
from model import model, ebar
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import cross_val_score

input_dir = "../sample_data" # A remplacer par public_data
output_dir = "../res"

basename = 'credit'
D = DataManager(basename, input_dir) # Load data
print D

mymodel = model()
 
# Train
Yonehot_tr = D.data['Y_train']
# Attention pour les utilisateurs de problemes multiclasse,
# mettre convert_to_num DANS la methode fit car l'ingestion program
# fournit Yonehot_tr a la methode "fit"
# Ceux qui resolvent des problemes a 2 classes ou des problemes de
# regression n'en ont pas besoin
Ytrue_tr = convert_to_num(Yonehot_tr, verbose=False) # For multi-class only, to be compatible with scikit-learn
mymodel = mymodel.fit(D.data['X_train'], Ytrue_tr)

# Making predictions
Ypred_tr = mymodel.predict(D.data['X_train'])
Ypred_va = mymodel.predict(D.data['X_valid'])
Ypred_te = mymodel.predict(D.data['X_test'])  

# We can compute the training success rate 
acc_tr = accuracy_score(Ytrue_tr, Ypred_tr)
# But it might be optimistic compared to the validation and test accuracy
# that we cannot compute (except by making submissions to Codalab)
# So, we can use cross-validation:
acc_cv = cross_val_score(mymodel, D.data['X_train'], Ytrue_tr, cv=5, scoring='accuracy')

print "One sigma error bars:"
print "Training Accuracy = %5.2f +-%5.2f" % (acc_tr, ebar(acc_tr, Ytrue_tr.shape[0]))
print "Cross-validation Accuracy = %5.2f +-%5.2f" % (acc_cv.mean(), acc_cv.std())

# If you want to be more conservative and use a 95% confidence interval, use 2*sigma                                          
