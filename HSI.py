# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 16:20:56 2020
This file is to save the information about HSI
@author: 22594658
"""

from scipy import io, misc

import numpy as np
import os
import spectral
from sklearn import preprocessing


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
    elif ext == '.npy':
        return np.load(dataset, allow_pickle=True)
    else:
        raise ValueError("Unknown file format: {}".format(ext))


class HSI():
    """ Generic class for a hyperspectral scene """

    def __init__(self, **hyperparams):

        super(HSI, self).__init__()
        self.name = hyperparams['dataset']
        self.norm_type = hyperparams['norm_type']

        self.folder = str(hyperparams['folder']) + str(self.name) + '/'
        print(self.folder)
        self.ignored_labels = []


        if self.name == 'co':
            # Load the data

            self.folder = str(hyperparams['folder']) + 'salt_moghimi' + '/'
            data = open_file(self.folder + 'co(CS).mat')
            data = data["data"]
            img = data[:, 0:-1]
            label = data[:, -1]

            self.img_channels = img.shape[1]

            self.gt = label
            self.gt = self.gt.reshape(self.gt.shape[0])
            self.gt = self.gt + 1
            self.gt = self.gt.astype(np.uint8)

            self.label_values = ["salt", "control"]
            self.category = 2


            self.ignored_labels = [0]

        elif self.name == 'Kharchia':
            # Load the data

            self.folder = str(hyperparams['folder']) + 'salt_moghimi' + '/'
            data = open_file(self.folder + 'Kharchia.mat')
            data = data["data"]
            img = data[:, 0:-1]
            label = data[:, -1]

            self.img_channels = img.shape[1]

            self.gt = label
            self.gt = self.gt.reshape(self.gt.shape[0])
            self.gt = self.gt + 1
            self.gt = self.gt.astype(np.uint8)

            self.label_values = ["salt", "control"]
            self.category = 2


            self.ignored_labels = [0]

        elif self.name == 'sp':
            # Load the data

            self.folder = str(hyperparams['folder']) + 'salt_moghimi' + '/'
            data = open_file(self.folder + 'sp(CS).mat')
            data = data["data"]
            img = data[:, 0:-1]
            label = data[:, -1]

            self.img_channels = img.shape[1]

            self.gt = label
            self.gt = self.gt.reshape(self.gt.shape[0])
            self.gt = self.gt + 1
            self.gt = self.gt.astype(np.uint8)

            self.label_values = ["salt", "control"]
            self.category = 2


            self.ignored_labels = [0]

        elif self.name == 'cs':
            # Load the data

            self.folder = str(hyperparams['folder']) + 'salt_moghimi' + '/'
            data = open_file(self.folder + 'CS.mat')
            data = data["data"]
            img = data[:, 0:-1]
            label = data[:, -1]

            self.img_channels = img.shape[1]

            self.gt = label
            self.gt = self.gt.reshape(self.gt.shape[0])
            self.gt = self.gt + 1
            self.gt = self.gt.astype(np.uint8)

            self.label_values = ["salt", "control"]
            self.category = 2


            self.ignored_labels = [0]


        # filter NaN out, if there is any none
        nan_mask = np.isnan(img.sum(axis=-1))
        if np.count_nonzero(nan_mask) > 0:
            print("Warning: NaN has been found in this dataset and the Nan mask will be disabled")

        img[nan_mask] = 0
        self.gt[nan_mask] = 0
        self.ignored_labels.append(0)

        self.ignored_labels = list(set(self.ignored_labels))
        # normalization
        img = np.asarray(img, dtype='float32')
        self.img = img


    def Normalize(self, norm_type):
        """
        This method is used to compute the normalization based on the normalization type
        """

        if norm_type is None:
            norm_type = 'normal'

        if norm_type == 'scale':
            # this process is the same with standarization
            # process the standarization
            # this process make the data to have mean=0 and std=1
            print("norm_type is scale")
            data = self.img.reshape(np.prod(self.img.shape[:2]),
                                    np.prod(self.img.shape[2:]))  # reshape data from 145*145*200 into 21025*200
            data = preprocessing.scale(data)
            self.img = data.reshape(self.img.shape[0], self.img.shape[1], self.img.shape[2])

        elif norm_type == 'normal':
            # process with zero one normalization
            self.img = (self.img - np.min(self.img)) / (np.max(self.img) - np.min(self.img))

        else:
            print("The normalization type you choose is not available")

        return self.img



