#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 19:17:29 2020

This code is to create K fold
@author: research
"""

#import numpy as np
import sklearn.model_selection 

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptc

from scipy import io, misc
import os
import spectral

from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit

def open_file(dataset):
    _, ext = os.path.splitext(dataset)
    ext = ext.lower()
    if ext == '.mat':
        # Load Matlab array
        return io.loadmat(dataset)
    elif ext == '.tif' or ext == '.tiff':
        # Load TIFF file
        return misc.imread(dataset)
    elif ext == '.hdr':
        img = spectral.open_image(dataset)
        return img.load()
    elif ext=='.npy':
        return np.load(dataset, allow_pickle=True)
    else:
        raise ValueError("Unknown file format: {}".format(ext))

def indexToAssignment(index_, Row, Col):
    new_assign = {}
    for counter, value in enumerate(index_):
        assign_0 = value // Col 
        assign_1 = value % Col 
        new_assign[counter] = [assign_0, assign_1]
    return new_assign


def create_kfold(name,k,size):
    
    flag=1 
    UseGroup=0


    if name=='cs':
        flag=0
        #print ("test")
        folder='../Datasets/salt_moghimi/'
        data = open_file(folder + 'CS.mat')
        
        #print(data)
        data = data["data"]
        img=data[:,0:-1]

        label=data[:,-1]
        data=img
        
        gt=label
        gt=gt+1
    elif name=='co':
        flag=0
        #print ("test")
        folder='../Datasets/salt_moghimi/'
        data = open_file(folder + 'co(CS).mat')
        
        #print(data)
        data = data["data"]
        img=data[:,0:-1]

        label=data[:,-1]
        data=img
        
        gt=label
        gt=gt+1
    elif name=='Kharchia':
        flag=0
        #print ("test")
        folder='../Datasets/salt_moghimi/'
        data = open_file(folder + 'Kharchia.mat')
        
        #print(data)
        data = data["data"]
        img=data[:,0:-1]

        label=data[:,-1]
        data=img
        
        gt=label
        gt=gt+1        
    elif name=='sp':
        flag=0
        #print ("test")
        folder='../Datasets/salt_moghimi/'
        data = open_file(folder + 'sp(CS).mat')
        
        #print(data)
        data = data["data"]
        img=data[:,0:-1]

        label=data[:,-1]
        data=img
        
        gt=label
        gt=gt+1 


    elif name=='Cassava2':
        flag = 0
        UseGroup = 1
        folder='/media/research/New Volume/wija/Python/Datasets/cassava_dataset2/'
        all_data = open_file(folder+'Cassava2.npy')[0]
        group = open_file(folder+'Group.npy')[0]
        data = all_data[:, 0:-1]
        gt = all_data [:, -1] 
        
        
    if UseGroup ==0:
        sss = StratifiedShuffleSplit(n_splits=k, train_size=size, random_state=345)
        if flag==1:
            X=data.reshape(data.shape[0]*data.shape[1],data.shape[2])
            Y=gt.reshape(gt.shape[0]*gt.shape[1])
        else:
            X=data
            Y=gt
            
        
        indices=np.nonzero(gt) #this means the unused label is removed
        indices=list(zip(*indices))
    
        #save the train and test indices into file 
        print ("X shape: ", X.shape)
        print ("Y shape: ", Y.shape)
        i=0   
        for train_index, test_index in sss.split(X, Y):
            if flag==1:
                training_indices=np.unravel_index(train_index,(data.shape[0],data.shape[1]))
                training_indices=list(zip(*training_indices))
                training_fix=list(set(training_indices) & set(indices))
                
                testing_indices=np.unravel_index(test_index,(gt.shape[0],gt.shape[1]))
                testing_indices=list(zip(*testing_indices))
                testing_fix=list(set(testing_indices)&set(indices))
            else:
                
                training_indices=train_index
                training_indices=training_indices.tolist()
                training_fix= [(i, ) for i in training_indices] 
                
                testing_indices=test_index
                testing_indices=testing_indices.tolist()
                testing_fix=[(i, ) for i in testing_indices]          
                
                #training_fix=train_index
                #testing_fix=test_index
            
            with open('/media/research/New Volume/wija/Python/Datasets/Data/train_test_'+name+'_'+str(i)+'.pkl','wb') as f:
                pickle.dump([training_fix,testing_fix],f)

            i=i+1
    else: #example for cassava dataset where it has group
        
        sss = GroupShuffleSplit(n_splits=k, random_state=345)
        if flag==1:
            X=data.reshape(data.shape[0]*data.shape[1],data.shape[2])
            Y=gt.reshape(gt.shape[0]*gt.shape[1])
        else:
            X=data
            Y=gt
            
        
        indices=np.nonzero(gt) #this means the unused label is removed
        indices=list(zip(*indices))
    
        #save the train and test indices into file 
        print ("X shape: ", X.shape)
        print ("Y shape: ", Y.shape)
        i=0   
        for train_index, test_index in sss.split(X, Y, groups = group):
            if flag==1:
                training_indices=np.unravel_index(train_index,(data.shape[0],data.shape[1]))
                training_indices=list(zip(*training_indices))
                training_fix=list(set(training_indices) & set(indices))
                
                testing_indices=np.unravel_index(test_index,(gt.shape[0],gt.shape[1]))
                testing_indices=list(zip(*testing_indices))
                testing_fix=list(set(testing_indices)&set(indices))
            else:
                
                training_indices=train_index
                training_indices=training_indices.tolist()
                training_fix= [(i, ) for i in training_indices] 
                
                testing_indices=test_index
                testing_indices=testing_indices.tolist()
                testing_fix=[(i, ) for i in testing_indices]          
                
                #training_fix=train_index
                #testing_fix=test_index
            
            with open('/media/research/New Volume/wija/Python/Datasets/Data/train_test_'+name+'_'+str(i)+'.pkl','wb') as f:
                pickle.dump([training_fix,testing_fix],f)

            i=i+1
            




def open_kfold(name,i):
    
    data_name='/media/research/New Volume/wija/Python/Datasets/Data/train_test_'+name+'_'+str(i)+'.pkl'
    
    with open(data_name,'rb') as f:
        train_index,test_index=pickle.load(f)
    
    return train_index,test_index


    
#create_kfold_equally('Fusarium',10,75828)
            
#train_index,test_index=open_kfold('Fusarium',0)        
 
#create_kfold('Cassava2',5,0)
#train_index,test_index=open_kfold('Cassava2',0)
      
#create_disjoint2('IndianPines',0.5)

#train_index,test_index=open_kfold('alldata',0)   

#draw_separate_train_test('IndianPines',0)

#train_indices,test_indices=create_disjoint_368('IndianPines')
#draw_separate_train_test('IndianPines',0)
        

