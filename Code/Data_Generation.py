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

sys.path.append('..')

# Built-in/Generic Imports

__author__ = '{Chao ZHOU}'
__copyright__ = 'Copyright {02/01/2020}, {Variational autoencoder}'
__email__ = '{chaozhouucl@gmail.com}'
__status__ = '{dev_status}'


# {code}
# example of a bimodal constructed from two gaussian processes

def gmm_data_generation(NO_DATA):
    mix_coef = [0.3, 0.7]
    X1 = normal(loc=10, scale=10, size=int(NO_DATA * mix_coef[0]))
    X2 = normal(loc=40, scale=6, size=NO_DATA - int(NO_DATA * mix_coef[0]))
    X = hstack((X1, X2))
    return X
