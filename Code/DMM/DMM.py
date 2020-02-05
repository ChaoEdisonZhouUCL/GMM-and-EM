"""
Description:
    this is the module provides EM algorithm for DMM.

"""

# Futures
from __future__ import print_function

import os
import copy
from scipy.stats import norm
from Code.Modules.Data_Generation import *
from Code.Modules.utils import *

sys.path.append('..')

# Built-in/Generic Imports

__author__ = '{Chao ZHOU}'
__copyright__ = 'Copyright {04/02/2020}, {EM algorithm study}'
__email__ = '{chaozhouucl@gmail.com}'
__status__ = '{dev_status}'


# {code}
def main():
    # ------------------------------------------------------------------------------------------------------------------
    # define project path
    # ------------------------------------------------------------------------------------------------------------------
    project_path = os.path.abspath(os.path.join(os.path.join(os.getcwd(), '..'), '..')) + '/'

    # ------------------------------------------------------------------------------------------------------------------
    # generate synthetic data and visualize them
    # ------------------------------------------------------------------------------------------------------------------
    # GMM data generation
    NO_DATA = 100000
    True_dist_param = {
        'dir_param_1': [6, 25, 9],
        'dir_param_2': [7, 8, 23],
        'mix_coef': [0.3, 0.7]
    }
    x, label = dmm_data_generation(NO_DATA, True_dist_param)

    # GMM data visualization
    data_path = project_path + 'Data/GMM/'
    plt.savefig(data_path + 'visualization of data.png', format='png')
    plt.close()

    # ------------------------------------------------------------------------------------------------------------------
    # initialize a GMM model with known number of cluster=2
    # ------------------------------------------------------------------------------------------------------------------
    NO_cluster = 2
    init_Mix_coef = np.asarray([0.5, 0.5])
    init_gaussain1_param = {'gaussian1_mean': 1, 'guassian1_var': 3}
    init_gaussain2_param = {'gaussian2_mean': 50, 'guassian2_var': 3, }
    init_respons = np.zeros(shape=(NO_DATA, NO_cluster))

    new_Mix_coef = copy.deepcopy(init_Mix_coef)
    old_Mix_coef = copy.deepcopy(init_Mix_coef)

    new_gaussain1_param = copy.deepcopy(init_gaussain1_param)
    old_gaussain1_param = copy.deepcopy(init_gaussain1_param)

    new_gaussain2_param = copy.deepcopy(init_gaussain2_param)
    old_gaussain2_param = copy.deepcopy(init_gaussain2_param)

    new_respons = copy.deepcopy(init_respons)
    old_respons = copy.deepcopy(init_respons)

    # ------------------------------------------------------------------------------------------------------------------
    # run EM for 10 times
    # ------------------------------------------------------------------------------------------------------------------
    for i in range(10):
        new_respons = EM_E(x, new_respons, new_gaussain1_param, new_gaussain2_param, new_Mix_coef)
        new_gaussain1_param, new_gaussain2_param, new_Mix_coef = EM_M(x, new_respons, new_gaussain1_param,

                                                                      new_gaussain2_param, new_Mix_coef)
    est__dist_param = {'gaussian1_params': new_gaussain1_param,
                       'gaussian2_params': new_gaussain2_param,
                       'mix_coef': new_Mix_coef
                       }
    results_path = project_path + 'Results/GMM/'
    write_data(True_dist_param, file_path=results_path + 'True_dist_params.pkl')
    write_data(est__dist_param, file_path=results_path + 'Est_dist_params.pkl')


# E-step
def EM_E(x, new_respons, new_gaussain1_param, new_gaussain2_param, new_Mix_coef):
    NO_cluster = len(new_Mix_coef)
    old_respons = copy.deepcopy(new_respons)
    gaussian1 = norm(loc=new_gaussain1_param['gaussian1_mean'],
                     scale=new_gaussain1_param['guassian1_var'])
    gaussian2 = norm(loc=new_gaussain2_param['gaussian2_mean'],
                     scale=new_gaussain2_param['guassian2_var'])
    gaussian_mixture = [gaussian1, gaussian2]
    for k in range(NO_cluster):
        pi = new_Mix_coef[k]
        gaussian_pdf = gaussian_mixture[k].pdf(x)
        new_respons[:, k] = pi * gaussian_pdf

    new_respons = new_respons / np.sum(new_respons, axis=1, keepdims=True)
    print('new respons:\r\n')
    print(new_respons)
    return new_respons


# M-Step
def EM_M(x, new_respons, new_gaussain1_param, new_gaussain2_param, new_Mix_coef):
    NO_DATA = len(x)
    old_gaussain1_param = copy.deepcopy(new_gaussain1_param)
    old_gaussain2_param = copy.deepcopy(new_gaussain2_param)
    old_Mix_coef = copy.deepcopy(new_Mix_coef)

    new_Mix_coef = np.sum(new_respons, axis=0, keepdims=True) / NO_DATA
    new_Mix_coef = np.squeeze(new_Mix_coef)

    new_gaussain1_param['gaussian1_mean'] = np.dot(new_respons[:, 0].T, x) / np.sum(new_respons[:, 0], axis=0)
    new_gaussain2_param['gaussian2_mean'] = np.dot(new_respons[:, 1].T, x) / np.sum(new_respons[:, 1], axis=0)

    new_gaussain1_param['guassian1_var'] = np.sqrt(np.dot(new_respons[:, 0].T,
                                                          np.power(x - new_gaussain1_param['gaussian1_mean'],
                                                                   2)) / \
                                                   np.sum(new_respons[:, 0], axis=0))
    new_gaussain2_param['guassian2_var'] = np.sqrt(np.dot(new_respons[:, 1].T,
                                                          np.power(x - new_gaussain2_param['gaussian2_mean'],
                                                                   2)) / \
                                                   np.sum(new_respons[:, 1], axis=0))
    print('new mix coeff:\r\n')
    print(new_Mix_coef)

    print('new gaussian1 params:\r\n')
    print(new_gaussain1_param)

    print('new gaussian2 params:\r\n')
    print(new_gaussain2_param)
    return new_gaussain1_param, new_gaussain2_param, new_Mix_coef


if __name__ == '__main__':
    main()
