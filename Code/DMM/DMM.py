"""
Description:
    this is the module provides EM algorithm for DMM.

"""

# Futures
from __future__ import print_function

import os
import copy
from scipy.stats import dirichlet
from scipy import optimize
from scipy.special import digamma, polygamma
from Code.Modules.Data_Generation import *
from Code.Modules.utils import *
from mpl_toolkits.mplot3d import Axes3D

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
    # DMM data generation
    NO_DATA = 100000
    True_dist_param = {
        'dir_param_1': [6, 25, 9],
        'dir_param_2': [7, 8, 23],
        'mix_coef': [0.3, 0.7]
    }
    x, label = dmm_data_generation(NO_DATA, True_dist_param)

    # DMM data save
    data_path = project_path + 'Data/DMM/'
    write_data(x, file_path=data_path + 'DMM_Samples.pkl')
    write_data(label, file_path=data_path + 'DMM_Samples_Label.pkl')

    # DMM data visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(111)
    ax.scatter(
        # x[np.where(label == 1)[0], 0],
        x[np.where(label == 1)[0], 1],
        x[np.where(label == 1)[0], 2],
        c='g',
        label='Dir1',
        alpha=0.5)
    ax.scatter(
        # x[np.where(label == 2)[0], 0],
        x[np.where(label == 2)[0], 1],
        x[np.where(label == 2)[0], 2],
        c='b',
        label='Dir2',
        alpha=0.5)
    plt.legend()
    plt.savefig(data_path + 'visualization of data.png', format='png')
    plt.close()

    # ------------------------------------------------------------------------------------------------------------------
    # initialize a DMM model with known number of cluster=2
    # ------------------------------------------------------------------------------------------------------------------
    NO_cluster = 2
    init_Mix_coef = np.asarray([0.5, 0.5])
    init_dir1_param = np.asarray([5, 20, 10])
    init_dir2_param = np.asarray([8, 7, 20])
    init_respons = np.zeros(shape=(NO_DATA, NO_cluster))

    new_Mix_coef = copy.deepcopy(init_Mix_coef)
    old_Mix_coef = copy.deepcopy(init_Mix_coef)

    new_dir1_param = copy.deepcopy(init_dir1_param)
    old_dir1_param = copy.deepcopy(init_dir1_param)

    new_dir2_param = copy.deepcopy(init_dir2_param)
    old_dir2_param = copy.deepcopy(init_dir2_param)

    new_respons = copy.deepcopy(init_respons)
    old_respons = copy.deepcopy(init_respons)

    # ------------------------------------------------------------------------------------------------------------------
    # run EM for 10 times
    # ------------------------------------------------------------------------------------------------------------------
    for i in range(10):
        new_respons = EM_E(x, new_respons, new_dir1_param, new_dir2_param, new_Mix_coef)
        new_dir1_param, new_dir2_param, new_Mix_coef = EM_M(x, new_respons, new_dir1_param, new_dir2_param,
                                                            new_Mix_coef)
    est__dist_param = {'dir1_params': new_dir1_param,
                       'dir2_params': new_dir2_param,
                       'mix_coef': new_Mix_coef
                       }
    results_path = project_path + 'Results/DMM/'
    write_data(True_dist_param, file_path=results_path + 'True_dist_params.pkl')
    write_data(est__dist_param, file_path=results_path + 'Est_dist_params.pkl')


# E-step
def EM_E(x, new_respons, new_dir1_param, new_dir2_param, new_Mix_coef):
    NO_cluster = len(new_Mix_coef)
    old_respons = copy.deepcopy(new_respons)
    dirichlet1 = dirichlet(alpha=new_dir1_param)
    dirichlet2 = dirichlet(alpha=new_dir2_param)
    dirichlet_mixture = [dirichlet1, dirichlet2]
    for k in range(NO_cluster):
        pi = new_Mix_coef[k]
        dirichlet_pdf = dirichlet_mixture[k].pdf(x.T)
        new_respons[:, k] = pi * dirichlet_pdf

    new_respons = new_respons / np.sum(new_respons, axis=1, keepdims=True)
    print('new respons:\r\n')
    print(new_respons)
    return new_respons


# M-Step
def EM_M(x, new_respons, new_dir1_param, new_dir2_param, new_Mix_coef):
    NO_DATA = len(x)
    old_dir1_param = copy.deepcopy(new_dir1_param)
    old_dir2_param = copy.deepcopy(new_dir2_param)
    old_Mix_coef = copy.deepcopy(new_Mix_coef)

    new_Mix_coef = np.sum(new_respons, axis=0, keepdims=True) / NO_DATA
    new_Mix_coef = np.squeeze(new_Mix_coef)

    new_dir1_param = digamma(np.sum(old_dir1_param)) + np.dot(new_respons[:, 0].T, np.log(x)) / np.sum(
        new_respons[:, 0])
    for id, value in enumerate(new_dir1_param):
        new_dir1_param[id] = inv_digamma(value)

    new_dir2_param = digamma(np.sum(old_dir2_param)) + np.dot(new_respons[:, 1].T, np.log(x)) / np.sum(
        new_respons[:, 1])
    for id, value in enumerate(new_dir2_param):
        new_dir2_param[id] = inv_digamma(value)

    print('new mix coeff:\r\n')
    print(new_Mix_coef)

    print('new dir1 params:\r\n')
    print(new_dir1_param)

    print('new dir2 params:\r\n')
    print(new_dir2_param)
    return new_dir1_param, new_dir2_param, new_Mix_coef


# calc inverse digamma
def inv_digamma(y, eps=1e-8, max_iter=100):
    '''
    Numerical inverse to the digamma function by root finding;
    this can only handle scalar, not matrix.
    '''

    if y >= -2.22:
        xold = np.exp(y) + 0.5
    else:
        xold = -1 / (y - digamma(1))

    for _ in range(max_iter):

        xnew = xold - (digamma(xold) - y) / polygamma(1, xold)

        if xnew <= 0:
            xnew = xold

        if np.abs(xold - xnew) < eps:
            break

        xold = xnew

    return xnew


if __name__ == '__main__':
    main()
