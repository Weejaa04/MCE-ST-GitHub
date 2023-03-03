# -*- coding: utf-8 -*-
"""
Created on Thursday, 17 March 2022

This code is work in pytorch

This code consist:
  
@author: Wijayanti Nurul Khotimah (22594658)
"""

# Python 2/3 compatiblity
from __future__ import print_function
from __future__ import division

# for the argument

import argparse
import numpy as np

# Torch
import torch
import torch.utils.data as datatorch  # what is it for?
from torchsummary import summary

# connecting the class
from HSI import *
from experiment import Experiment
from train import *
import math
from MyHyperX import *
from sklearn import metrics, preprocessing
import random

from utils2 import weights_init

from sklearn.decomposition import PCA


# argument parser for interaction

def get_dinamic_bacth(size, n):
    modulus = size % n
    #print ("size: ", size)
    #print ("n: ", n)
    #print ("modulus: ", modulus)

    if modulus < 64: #64
        result = int(size / n)

        if result == 0:
            new_batch_size = n
        else:
            new_batch_size = n + math.ceil(modulus / result)
    else:
        new_batch_size = n

    return new_batch_size


def compute_weights (counts, B, nCategory):
    total = sum (counts)
    counts = counts/total
    effective_num = 1.0 - np.power(B, counts)
    weights = (1.0 - B) / np.array(effective_num)
    weights = weights / np.sum(weights) * nCategory #normalize the weighs, make their total is the same with the number of class
    
    
    return weights
    


parser = argparse.ArgumentParser(
    description="Run deep learning experiments on various hyperspectral datasets with various method")
parser.add_argument('--dataset', type=str,  help="Dataset name to use.")
parser.add_argument('--model', type=str, default="MCE-ST",
                    help="Model to train. Available:\n"
                         "ssrn (based on paper 25), "
                         "mou (1D RNN)")
parser.add_argument('--folder', type=str, help="dataset location "
                                               "datasets (defaults to the current working directory).",
                    default="Datasets/")
parser.add_argument('--cuda', type=int, default=-1,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")
parser.add_argument('--runs', type=int, default=1, help="Number of runs (default: 1)")

# Dataset options
group_dataset = parser.add_argument_group('Dataset')
group_dataset.add_argument('--training_sample', type=float, default=10,
                           help="Percentage of samples to use for training (default: 10%)")
# if the percentage is between 0 and 1, so it was in percent, but if more than one mean the number of training in each class
group_dataset.add_argument('--sampling_mode', type=str, help="Sampling mode"
                                                             " (random sampling or disjoint, default: random)",
                           default='random')
group_dataset.add_argument('--norm_type', type=str, default='scale', help="the mode of the training")
# Training options
group_train = parser.add_argument_group('Training')
group_train.add_argument('--epoch', type=int,default=200, help="Training epochs (optional, if"
                                                   " absent will be set by the model)")
group_train.add_argument('--patch_size', type=int, 
                         help="Size of the spatial neighbourhood (optional, if "
                              "absent will be set by the model)")
group_train.add_argument('--lr', type=float, default=0.0002,
                         help="Learning rate, set by the model if not specified.")
group_train.add_argument('--batch_size', type=int, default = 256,
                         help="Batch size (optional, if absent will be set by the model")
group_train.add_argument('--kernel_size', type=int,
                         help="Kernel size (optional, if absent will be set by the model")
group_train.add_argument('--padded', type=int, default=1,
                         help="the data is padded or not (the default is one, which means yes)")
group_train.add_argument('--val_size', type=int, default=0.1,
                         help="validation size (optional, if absent will be set by the model")
group_model = parser.add_argument_group('Model')
group_model.add_argument('--dmodel', type=int, default=96, help="the size of feature vector")
group_model.add_argument('--nhead', type=int, default=4, help="the number of head in MHA")
group_model.add_argument('--depth', type=int, default=5, help="the number of encoder")
group_model.add_argument('--dropout', type=float, default=0.1)


args = parser.parse_args()


if args.dataset != None:
    hyperparams = vars(args)  # the type of hyperparams is dictionary
    print("args is null ")
else:
    hyperparams = {
        "dataset": "co",  # this is for salt stress dataset
        "folder": "Datasets/",
        "model": "MCE-ST", #for my method: MySpectralSqueezeTransformer10
        "epoch": 200,
        "runs": 1,
        "flag": 0,
        "training_sample": 0.3,
        "val_size": 0,
        "sampling_mode": "kfold",
        "lr": 0.0002,
        "patch_size": 17,
        "batch_size": 256,
        "ignored_labels": 0,
        "norm_type": "scale",
        "padded": 1,
        "dmodel": 96,
        "nhead": 4,
        "depth": 5,
        "dropout": 0.1,
        "ignored_labels": [0],
        "supervision": "full",
        "device": "cuda",
        "Beta": 1, #this is wor class weight loss for imbalance data 
        "channels": 1,
        "restore": None,
    }
# Receive the argument

sampling_mode = hyperparams['sampling_mode']
training_sample = hyperparams['training_sample']

print ("batch size:", hyperparams["batch_size"])

supervision = 'full'  # at first we make the supervision is full inorder to make the framework can work with semisupervised and unsupervised
adaptive = 1

# Print to check whether the argument is working


HSIs = HSI(**hyperparams)

ITER = hyperparams['runs']

print("HSIs name: " + HSIs.name)

# default ITER = 1 --> if we used k fold then ITER = k

########the end of the part###################

seeds = [1334]
patience = 200
total_OA = 0
total_AA = 0
total_kappa = 0
e_A = np.zeros(HSIs.category)
total_f_mean=0
total_f1=np.zeros(HSIs.category)


listOA = []
listAA = []
listkappa = []
listEA = []
ListTrainTime = []
ListTestTime = []
listF1=[]
ListFmean=[]

index_range=0
for index_range in range(index_range,index_range+ITER):

    print("Iteration: {}".format(index_range + 1))

    # np.random.seed(seeds)
    hyper = hyperparams

    Ex = Experiment(HSIs.gt, **hyperparams)

    Ex.set_train_test(HSIs.gt, index_range)  # set the training and testing
    # Ex.set_train_test_val(HSIs.gt, HSIs.traing_standard) #set the training and testing

    if hyperparams['model'] != 'AE':
        y_train = HSIs.gt[tuple(Ex.training_indices)] - 1  # reduce each indices with 1 --> this is 1D array
        #print("y_train size: ", y_train.shape)
        y_test = HSIs.gt[tuple(Ex.testing_indices)] - 1
        #print("y_test size: ", y_test.shape)
        y_test_save = y_test
    ###this code only to check if not using argument
    
    #print ("Model: ", hyperparams['model'])
    if hyperparams['model']=='MCE-ST':
        #Note: SpectralTransformer is my Transformer where the token is based on spectralFormer
        data = preprocessing.scale(HSIs.img)
        data_spatial = data
        

        x_train= data_spatial[tuple(Ex.training_indices)]
        x_test = data_spatial[tuple(Ex.testing_indices)] 
        nband = x_train.shape[1]  # model dimension

        if adaptive == 1:
            #print ("len training indices: ", x_train.shape[0])
            new_batch_size = get_dinamic_bacth(x_train.shape[0], Ex.batch_size)
            print("batch_size", new_batch_size)
            Ex.batch_size = new_batch_size

        print ("nband :", nband)


        train_dataset = MyHyperX(x_train, y_train, **hyperparams)
        train_loader = datatorch.DataLoader(train_dataset, Ex.batch_size, shuffle=True)
        

        test_dataset = MyHyperX(x_test, y_test, **hyperparams)
        test_loader = datatorch.DataLoader(test_dataset, Ex.batch_size)




        # the process for compiling
        #print("compiling process")

        training = True

        if training:
            try:

                train(nband, train_loader, HSIs.category,index_range,**hyperparams)  # call train process
            except KeyboardInterrupt:
                pass

        #this code is for testing
        testing = True
        if testing:
            model_path = 'model/%s_%s_%d.pth'%(hyperparams["model"],hyperparams["dataset"],index_range) #here i start by initialize the model
            y_pred, y_test_save = evaluate_model(test_loader, model_path,hyperparams['model'])


            y_pred = np.argmax(y_pred, axis=1)


        
            overall_acc = metrics.accuracy_score(y_pred, y_test_save)
            print("Accuracy: ", overall_acc)
            #listOA.append(overall_acc)
            confusion_matrix = metrics.confusion_matrix(y_pred, y_test_save)
            each_acc, average_acc = Ex.get_Ave_Accuracy(confusion_matrix)
            kappa = metrics.cohen_kappa_score(y_pred, y_test_save)
            f_mean=metrics.f1_score( y_pred,y_test_save, average='macro')
            f1_=metrics.f1_score( y_pred,y_test_save, average=None)
            
    

    #if ITER > 1:
    listAA.append(average_acc)
    listEA.append(each_acc)
    listOA.append(overall_acc)
    listkappa.append(kappa)
    listF1.append(f1_)
    ListFmean.append(f_mean)


    total_OA = total_OA + overall_acc
    total_AA = total_AA + average_acc
    total_kappa = total_kappa + kappa
    e_A = e_A + each_acc
    


OA = total_OA / ITER
AA = total_AA / ITER
K = total_kappa / ITER
EA = e_A / ITER


print("OA: {}".format(OA))
print("AA: {}".format(AA))
print("Kappa: {}".format(K))
print("Each_Accuracy: {}".format(EA))

if ITER > 1:
    print("ListOA: {}".format(listOA))
    print("ListAA: {}".format(listAA))
    print("ListKappa: {}".format(listkappa))
    print("ListEach_Accuracy: {}".format(listEA))
    print ("List F-mean: {}".format(ListFmean))
    print ("List F-1: {}".format(listF1))
