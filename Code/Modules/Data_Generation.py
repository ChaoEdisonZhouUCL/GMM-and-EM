"""
Description:
    this is the module provides synthetic data generation code for GMM.

"""

# Futures
from __future__ import print_function

import sys

import matplotlib.pyplot as plt
from numpy import hstack
from numpy.random import normal
import numpy as np

sys.path.append('..')

# Built-in/Generic Imports

__author__ = '{Chao ZHOU}'
__copyright__ = 'Copyright {02/01/2020}, {Variational autoencoder}'
__email__ = '{chaozhouucl@gmail.com}'
__status__ = '{dev_status}'


# {code}
def gmm_data_generation(NO_DATA, True_dist_param):
    gaussian1_params = True_dist_param['gaussian1_params']
    gaussian_param_1_mean = gaussian1_params['gaussian_param_1_mean']
    gaussian_param_1_scale = gaussian1_params['gaussian_param_1_scale']

    gaussian2_params = True_dist_param['gaussian2_params']
    gaussian_param_2_mean = gaussian2_params['gaussian_param_2_mean']
    gaussian_param_2_scale = gaussian2_params['gaussian_param_2_scale']

    mix_coef = True_dist_param['mix_coef']

    NO_DATA_1 = int(NO_DATA * mix_coef[0])
    NO_DATA_2 = NO_DATA - NO_DATA_1
    X1 = normal(loc=gaussian_param_1_mean, scale=gaussian_param_1_scale, size=NO_DATA_1)
    X2 = normal(loc=gaussian_param_2_mean, scale=gaussian_param_2_scale, size=NO_DATA_2)
    X = hstack((X1, X2))
    return X


def dmm_data_generation(NO_DATA, True_dist_param):
    '''
    A Dirichlet mixture model data synthetic code.

    :param NO_DATA: int, the total number of data.
    :param LV_dist_param: dict, the true parameters of dirichlet mixture model, a structure like:
            {
            'dir_param_1': [6, 25, 9],
            'dir_param_2': [7, 8, 23],
            'mix_coef': [0.3, 0.7]
            }, indicating 2 dirichlet mixture with data_dim=3.
    :return: data, np array with shape=(NO_DATA, data_dim),shuffled data from DMM.
    :return: label, np array with shape=(NO_DATA, data_dim), shuffled class label for each data.
    '''
    ############################
    # parse True_dist_param
    ############################
    NO_DATA = NO_DATA
    NO_data_per_cluster = []
    mix_coef = True_dist_param['mix_coef']
    no_cluster = len(mix_coef)

    for k in range(no_cluster):
        if k + 1 == no_cluster:
            no_data = NO_DATA - np.sum(NO_data_per_cluster, dtype=np.int32)
        else:
            no_data = int(NO_DATA * mix_coef[k])
        NO_data_per_cluster.append(no_data)

        for key, value in True_dist_param.items():
            if str(k + 1) in key:
                dir_params = value
                break

        data = np.random.dirichlet(alpha=dir_params, size=no_data)
        label = np.ones(shape=(no_data, 1)) * (k + 1)
        data = np.concatenate((data, label), axis=1)
        if k == 0:
            DATA = data
        else:
            DATA = np.concatenate((DATA, data))

    np.random.shuffle(DATA)
    label = DATA[:, -1]
    data = DATA[:, :-1]

    return data, label
