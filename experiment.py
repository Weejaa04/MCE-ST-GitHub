# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 09:51:51 2020

@author: 22594658
"""

import random
import numpy as np


import matplotlib.pyplot as plt
from Utils import zeroPadding 
from operator import truediv
import pickle
from kfold import open_kfold
import sklearn.model_selection



def indexToAssignment(index_, Row, Col):
    new_assign = {}
    for counter, value in enumerate(index_):
        assign_0 = value // Col 
        assign_1 = value % Col 
        new_assign[counter] = [assign_0, assign_1]
    return new_assign



class Experiment():
    
    """
    This class is used as a framework for the experiment
    
    """
    
    def __init__(self,gt,**parameters):
        super(Experiment,self).__init__()
        
        self.training_sample=parameters['training_sample']
        self.sampling_mode=parameters['sampling_mode']
        self.name=parameters['dataset']

        self.padded=parameters['padded']
        self.val_sample=parameters['val_size']
        
        if self.sampling_mode is None:
            self.sampling_mode = 'random'
            
        self.patch_size=parameters['patch_size']
        
        if self.patch_size is None:
            self.patch_size = 7
            
        if self.padded is None:
            self.padded=1
            
        
        self.epoch=parameters['epoch']
        self.lr=parameters['lr']
        self.batch_size=parameters['batch_size']
        self.training_indices={} #save the indices of the training sample 
        self.testing_indices={} #save the indices of the testing sample

        self.val_indices={}
        self.training_standard={}
        
    def set_train_test(self,gt,i,num_train=None):
        indices=np.nonzero(gt) #this means the unused label is removed
        X=list(zip(*indices))
        y=gt[indices].ravel()
        

        
        if self.training_sample > 1:
            self.training_sample = int (self.training_sample)
        
        if self.sampling_mode == 'random':
            training_indices, testing_indices = sklearn.model_selection.train_test_split(X, train_size=self.training_sample,stratify=y)
            
            self.training_indices = [list(t) for t in zip(*training_indices)]
            self.testing_indices = [list(t) for t in zip(*testing_indices)]

        
        elif self.sampling_mode =='kfold':

            print ("i",i)
            training_indices, testing_indices=open_kfold(self.name,i)

            self.training_indices=[list(t) for t in zip (*training_indices)]
            self.testing_indices=[list(t) for t in zip (*testing_indices)]  

        
        
        elif self.sampling_mode=='fixed': #means the sampling mode fix per class
            print ("Sampling {} with train size = {}".format(self.sampling_mode,self.training_sample))
            train_indices, test_indices = [], []

            
            unique, counts=np.unique(gt,return_counts=True)
            
            for c in np.unique(gt):
                if c == 0 or counts[c]<self.training_sample:
                    continue
                indices = np.nonzero(gt == c)
                X = list(zip(*indices)) # x,y features

                train, test = sklearn.model_selection.train_test_split(X, train_size=self.training_sample)
                train_indices += train
                test_indices += test
            train_indices = [list(t) for t in zip(*train_indices)]
            test_indices = [list(t) for t in zip(*test_indices)]
 
            self.training_indices=train_indices
            self.testing_indices=test_indices
        
        elif self.sampling_mode=='standard': #divide the training and testing data based on the standard of GRSS DASE initiative
            print ("Sampling with train the standard data of GRSS DASE initiative")
            train_indices, test_indices = [], []
                      
            for c in np.unique(gt):
                if c == 0: 
                    continue
                indices = np.nonzero(gt == c)
                X = list(zip(*indices)) # x,y features

                train, test = sklearn.model_selection.train_test_split(X, train_size=num_train[c])
                train_indices += train
                test_indices += test
            train_indices = [list(t) for t in zip(*train_indices)]
            test_indices = [list(t) for t in zip(*test_indices)]

            self.training_indices=train_indices
            self.testing_indices=test_indices
        
        
        elif self.sampling_mode=='manual': # I try to create the sampling mode manual 
            print ("sampling mode manual")
            train = {}
            test = {}
            groundTruth=gt
            proptionVal=1-self.training_sample
            m = np.amax(groundTruth) #return the max of groundtruth (the number of class)
            for i in range(m):
                indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1] #return the index of class i+1
                np.random.shuffle(indices) #these indics is shuffled
                #labels_loc[i] = indices
                nb_val = int(proptionVal * len(indices)) #80% of indices
                train[i] = indices[:-nb_val] #consist the index of the training
                test[i] = indices[-nb_val:] #consist the indices of the testing

            train_indices = []
            test_indices = []
            for i in range(m):

                train_indices += train[i] #concate the training data
                test_indices += test[i] #concate the testing data
            np.random.shuffle(train_indices)
            np.random.shuffle(test_indices)
            train_assign = indexToAssignment(train_indices, gt.shape[0], gt.shape[1]) #train_assign is in dictionary
            train_assign_list=list(train_assign.values()) #the list version of train assign
            train_indices_test=[list(t) for t in zip(*train_assign_list)]
            self.training_indices=train_indices_test            
            
            test_assign = indexToAssignment(test_indices, gt.shape[0], gt.shape[1])
            test_assign_list=list(test_assign.values())
            self.testing_indices=[list(t) for t in zip(*test_assign_list)]


    def get_Ave_Accuracy(self, confusion_matrix):
        #counter = confusion_matrix.shape[0]
        list_diag = np.diag(confusion_matrix)
        list_raw_sum = np.sum(confusion_matrix, axis=1)
        each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
        average_acc = np.mean(each_acc)
        return each_acc, average_acc
    
