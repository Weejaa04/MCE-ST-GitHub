#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 13:11:34 2021

This class is for characterizes a dataset for PyTorch'
@author: research
"""


import torch
from sklearn.preprocessing import LabelEncoder


class MyHyperX(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_X, labels, **hyperparams):
        'Initialization'
        self.labels = labels
        self.list_X = list_X
        # label encode target and ensure the values are floats
        self.labels = LabelEncoder().fit_transform(self.labels)
        self.name = hyperparams['dataset']

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

  def __getitem__(self, index):
        'Generates one sample of data'

        # Load data and get label
        X = self.list_X[index]
        y = self.labels[index]

        return X, y
