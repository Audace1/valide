#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 15:22:19 2018

@author: enzo
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This is an example of predictive model program, we show how to combine
a predictive model and a preprocessor with a pipeline. 

IMPORTANT: keep calling your program model.py so the ingestion program that
runs on Codalab can find it.
"""

from sys import argv, path
import numpy as np
import pickle

path.append ("../scoring_program")    # Contains libraries you will need
path.append ("../ingestion_program")  # Contains libraries you will need
from prepro import Preprocessor

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

def ebar(score, sample_num):
    '''ebar calculates the error bar for the classification score (accuracy or error rate)
    for sample_num examples'''
    return np.sqrt(1.*score*(1-score)/sample_num)

class model(BaseEstimator):
    '''model: modify this class to create a predictor of
    your choice. This could be your own algorithm, of one for the scikit-learn
    models, for which you optimize hyper-parameters with cross-validation.'''
    def __init__(self):
        '''This method initializes the parameters. This is where you could replace
        RandomPredictor by something else'''
        self.clf = FancyPredictor()

    def fit(self, X, y):
        ''' This is the training method: parameters are adjusted with training data.'''
        self.clf = self.clf.fit(X, y)
        return self

    def predict(self, X):
        ''' This is called to make predictions on test data. Predicted classes are output.'''
        #return self.clf.predict(X)
        
        #comme on a une auc de 0.12, on essaie de prédire l'autre classe
        #print (self.clf.predict_proba(X)[:,1])
        return self.clf.predict_proba(X)[:,1]

    def predict_proba(self, X):
        ''' Similar to predict, but probabilities of belonging to a class are output.'''
        return self.clf.predict_proba(X)[:,1] # The classes are in the order of the labels returned by get_classes
        
    def get_classes(self):
        return self.clf.classes_
        
    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        self = pickle.load(open(path + '_model.pickle'))
        return self

#########################################   
# Here is now a whole bunch of examples #
#########################################


class RandomPredictor:
    ''' Make random predictions.'''	
    def __init__(self):
        self.target_num=0
        return
        
    def __repr__(self):
        print self.target_num
        return "RandomPredictor"

    def __str__(self):
        return "RandomPredictor"
	
    def fit(self, X, Y):
        if Y.ndim == 1:
            self.target_num=len(set(Y))
        else:
            self.target_num=Y.shape[1]
        return self
		
    def predict_proba(self, X):
        prob = np.random.rand(X.shape[0],self.target_num)
        return prob	
    
    def predict(self, X):
        prob = self.predict_proba(X)
        yhat = [np.argmax(prob[i,:]) for i in range(prob.shape[0])]
        return yhat

class BasicPredictor(BaseEstimator):
    '''BasicPredictor: modify this class to create a predictor of
    your choice. This could be your own algorithm, of one for the scikit-learn
    models, for which you choose the hyper-parameters.'''
    def __init__(self):
        '''This method initializes the parameters. This is where you could replace
        RandomForestClassifier by something else or provide arguments, e.g.
        RandomForestClassifier(n_estimators=100, max_depth=2)'''
        self.clf = RandomForestClassifier()

    def fit(self, X, y):
        ''' This is the training method: parameters are adjusted with training data.'''
        self.clf = self.clf.fit(X, y)
        return self

    def predict(self, X):
        ''' This is called to make predictions on test data. Predicted classes are output.'''
        return self.clf.predict(X)

    def predict_proba(self, X):
        ''' Similar to predict, but probabilities of belonging to a class are output.'''
        return self.clf.predict_proba(X) # The classes are in the order of the labels returned by get_classes
        
    def get_classes(self):
        return self.clf.classes_
        
    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        self = pickle.load(open(path + '_model.pickle'))
        return self
    
    
class FancyPredictor(BaseEstimator):
    '''FancyPredictor: This is a more complex example that shows how you can combine
    basic modules (you can create many), in parallel (by voting using ensemble methods)
    of in sequence, by using pipelines.'''
    def __init__(self):
        '''You may here define the structure of your model. You can create your own type
        of ensemble. You can make ensembles of pipelines or pipelines of ensembles.
        This example votes among two classifiers: BasicClassifier and a pipeline
        whose classifier is itself an ensemble of GaussianNB classifiers.'''
        fancy_classifier = Pipeline([
                ('preprocessing', Preprocessor()),
                ('classification', BaggingClassifier(base_estimator=RandomForestClassifier()))
                ])
        self.clf = VotingClassifier(estimators=[
                ('basic', BasicPredictor()), 
                ('fancy', fancy_classifier)], 
                voting='soft')   
        self.num_train_samples=0
        self.num_feat=1
        self.num_labels=1
        self.is_trained=False
        
        
    def fit(self, X, y):
        ''' This is the training method: parameters are adjusted with training data.'''
        self.num_train_samples = len(X)
        if X.ndim>1: self.num_feat = len(X[0])
        print("FIT: dim(X)= [{:d}, {:d}]".format(self.num_train_samples, self.num_feat))
        num_train_samples = len(y)
        if y.ndim>1: self.num_labels = len(y[0])
        print("FIT: dim(y)= [{:d}, {:d}]".format(num_train_samples, self.num_labels))
        if (self.num_train_samples != num_train_samples):
            print("ARRGH: number of samples in X and y do not match!")
        
        self.clf = self.clf.fit(X, y)
        self.is_trained=True
        return self

    def predict(self, X):
        ''' This is called to make predictions on test data. Predicted classes are output.'''
        return self.clf.predict(X)
        

    def predict_proba(self, X):
        ''' Similar to predict, but probabilities of belonging to a class are output.'''
        
        num_test_samples = len(X)
        if X.ndim>1: num_feat = len(X[0])
        print("PREDICT: dim(X)= [{:d}, {:d}]".format(num_test_samples, num_feat))
        if (self.num_feat != num_feat):
            print("ARRGH: number of features in X does not match training data!")
        print("PREDICT: dim(y)= [{:d}, {:d}]".format(num_test_samples, self.num_labels))
        
        return self.clf.predict_proba(X) # The classes are in the order of the labels returned by get_classes
        
    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        self = pickle.load(open(path + '_model.pickle'))
        return self












if __name__=="__main__":
    # We can use this to run this file as a script and test the Classifier
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = "../public_data" # A remplacer par public_data
        output_dir = "../results"
    else:
        input_dir = argv[1]
        output_dir = argv[2];
                         
    from sklearn.metrics import accuracy_score      
    # Interesting point: the M2 prepared challenges using sometimes AutoML challenge metrics
    # not scikit-learn metrics. For example:
    from libscores import bac_metric
    from libscores import auc_metric
                 
    from data_manager import DataManager 
    from data_converter import convert_to_num 
    
    basename = 'credit'
    D = DataManager(basename, input_dir) # Load data
    print D
    
    # Here we define 3 classifiers and compare them
    classifier_dict = {
            'Pipeline': Pipeline([('prepro', Preprocessor()), ('classif', FancyPredictor())]),
            'RandomPred': RandomPredictor(),
            'BasicPred': BasicPredictor(),
            'FancyPred': FancyPredictor()}
        
    
    print "Classifier\tAUC\tBAC\tACC\tError bar"
    for key in classifier_dict:
        myclassifier = classifier_dict[key]
 
        # Train
        Yonehot_tr = D.data['Y_train']
        # Attention pour les utilisateurs de problemes multiclasse,
        # mettre convert_to_num DANS la methode fit car l'ingestion program
        # fournit Yonehot_tr a la methode "fit"
        # Ceux qui resolvent des problemes a 2 classes ou des problemes de
        # regression n'en ont pas besoin
        Ytrue_tr = convert_to_num(Yonehot_tr, verbose=False) # For multi-class only, to be compatible with scikit-learn
        myclassifier.fit(D.data['X_train'], Ytrue_tr)
        
        # Some classifiers and cost function use a different encoding of the target
        # values called on-hot encoding, i.e. a matrix (nsample, nclass) with one at
        # the position of the class in each line (also called position code):
        #nclass = len(set(Ytrue_tr))
        #Yonehot_tr = np.zeros([Ytrue_tr.shape[0],nclass])
        #for i, item in enumerate(Ytrue_tr): Yonehot_tr[i,item]=1
    
        # Making classification predictions (the output is a vector of class IDs)
        Ypred_tr = myclassifier.predict(D.data['X_train'])
        Ypred_va = myclassifier.predict(D.data['X_valid'])
        Ypred_te = myclassifier.predict(D.data['X_test']) 
        
        # Making probabilistic predictions (each line contains the proba of belonging in each class)
        Yprob_tr = myclassifier.predict_proba(D.data['X_train'])
        Yprob_va = myclassifier.predict_proba(D.data['X_valid'])
        Yprob_te = myclassifier.predict_proba(D.data['X_test']) 
    
        # Training success rate and error bar:
        # First the regular accuracy (fraction of correct classifications)
        acc = accuracy_score(Ytrue_tr, Ypred_tr)
        print key
        print "précision"
        print acc
        # Then two AutoML challenge metrics, working on the other representation
        #auc = auc_metric(Yonehot_tr, Yprob_tr, task='multiclass.classification')
        #bac = bac_metric(Yonehot_tr, Yprob_tr, task='multiclass.classification')
        # Note that the AutoML metrics are rescaled between 0 and 1.
        
        #print "%s\t%5.2f\t%5.2f\t%5.2f\t(%5.2f)" % (key, auc, bac, acc, ebar(acc, Ytrue_tr.shape[0]))
        #print "The error bar is valid for Acc only"
        # Note: we do not know Ytrue_va and Ytrue_te
        # See modelTest for a better evaluation using cross-validation
        
    # Another useful tool is the confusion matrix
    from sklearn.metrics import confusion_matrix
    print "Confusion matrix for %s" % key
    print confusion_matrix(Ytrue_tr, Ypred_tr)
    # On peut aussi la visualiser, voir:
    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    # Voir aussi http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
