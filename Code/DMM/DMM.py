"""
Description:
    this is the module provides EM algorithm for DMM.

"""

# Futures
from __future__ import print_function

import copy

from scipy.special import digamma, polygamma
from scipy.stats import dirichlet

from Code.Modules.Data_Generation import *
from Code.Modules.Dirichlet import *
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
    data_path = project_path + 'Data/DMM/'
    results_path = project_path + 'Results/DMM/'
    # ------------------------------------------------------------------------------------------------------------------
    # generate synthetic data and visualize them
    # ------------------------------------------------------------------------------------------------------------------
    # DMM data generation
    NO_DATA = 100000
    NO_Iterations = 3
    True_dist_param = {
        'true_dir1_params': [6, 25, 9],
        'true_dir2_params': [7, 8, 23],
        'mix_coef': [0.3, 0.7]
    }
    NO_cluster = len(True_dist_param['mix_coef'])
    x, label = dmm_data_generation(NO_DATA, True_dist_param)

    # DMM data save

    write_data(x, file_path=data_path + 'DMM_Samples.pkl')
    write_data(label, file_path=data_path + 'DMM_Samples_Label.pkl')

    # DMM data visualization
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for k in range(NO_cluster):
        ax.scatter(
            x[np.where(label == k + 1)[0], 0],
            x[np.where(label == k + 1)[0], 1],
            label='Dir{}'.format(k + 1),
            alpha=0.5)

    plt.legend()
    plt.savefig(data_path + 'visualization of data.png', format='png')
    plt.close()

    # ------------------------------------------------------------------------------------------------------------------
    # initialize a DMM model with known number of cluster
    # ------------------------------------------------------------------------------------------------------------------
    init_dist_params = {
        'mix_coef': np.asarray([0.5, 0.5]),
        'dir1_params': np.asarray([5, 20, 10]),
        'dir2_params': np.asarray([8, 7, 20])
    }
    init_respons = np.zeros(shape=(NO_DATA, NO_cluster))

    # ------------------------------------------------------------------------------------------------------------------
    # run EM
    # ------------------------------------------------------------------------------------------------------------------
    likelihood = []
    new_dist_params = copy.deepcopy(init_dist_params)
    new_respons = copy.deepcopy(init_respons)
    for i in range(NO_Iterations):
        new_respons = EM_E(x, new_respons, new_dist_params)
        new_dist_params = EM_M(x, new_respons, new_dist_params)

        # calc likelihood
        likelihood.append(calc_log_likelihood(dists_params=new_dist_params, data=x))

        # plot the intermeidate DMM contour
        draw_dir_contour(new_dist_params, data=x, respon=new_respons, filepath=results_path + 'epoch{}/'.format(i + 1))

    # ------------------------------------------------------------------------------------------------------------------
    # after EM
    # ------------------------------------------------------------------------------------------------------------------
    # save results
    est__dist_param = new_dist_params

    write_data(True_dist_param, file_path=results_path + 'True_dist_params.pkl')
    write_data(est__dist_param, file_path=results_path + 'Est_dist_params.pkl')
    write_data(likelihood, file_path=results_path + 'likelihood.pkl')

    # plot likelihood
    plot_likelihoood(likelihood, filepath=results_path)


# E-step
def EM_E(x, new_respons, new_dist_params):
    new_dist_params = copy.deepcopy(new_dist_params)
    new_Mix_coef = new_dist_params.pop('mix_coef')

    NO_cluster = len(new_Mix_coef)
    dirichlet_mixture = [dirichlet(alpha=value) for value in new_dist_params.values()]
    for k in range(NO_cluster):
        pi = new_Mix_coef[k]
        dirichlet_pdf = dirichlet_mixture[k].pdf(x.T)
        new_respons[:, k] = pi * dirichlet_pdf

    new_respons = new_respons / np.sum(new_respons, axis=1, keepdims=True)
    print('new respons:\r\n')
    print(new_respons)
    return new_respons


# M-Step
def EM_M(x, new_respons, new_dist_params):
    NO_DATA, NO_cluster = new_respons.shape
    old_dist_params = copy.deepcopy(new_dist_params)
    new_dist_params = copy.deepcopy(new_dist_params)

    # update pi
    new_Mix_coef = np.sum(new_respons, axis=0, keepdims=True) / NO_DATA
    new_Mix_coef = np.squeeze(new_Mix_coef)
    new_dist_params['mix_coef'] = new_Mix_coef

    # update the dir params
    k = 0
    for key, value in old_dist_params.items():
        if 'param' in key:
            old_dir_params = value
            new_dir_params = digamma(np.sum(old_dir_params)) + np.dot(new_respons[:, k].T, np.log(x)) / np.sum(
                new_respons[:, k])
            for id, param in enumerate(new_dir_params):
                new_dir_params[id] = inv_digamma(param)

            new_dist_params[key] = new_dir_params
            k += 1

    print('new_dist_params:\r\n')
    print(new_dist_params)

    return new_dist_params


# calc inverse digamma, for updating of params for dirichlet in M step
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


# calc the likelihood
def calc_log_likelihood(dists_params, data):
    dists_params = copy.deepcopy(dists_params)
    data = copy.deepcopy(data)
    mix_coef = dists_params.pop('mix_coef')
    no_cluster = len(mix_coef)
    likelihood = np.zeros(shape=(data.shape[0], no_cluster))

    for k in range(no_cluster):
        pi = mix_coef[0]
        for key, dir_param in dists_params.items():
            if str(k + 1) in key:
                dir = dirichlet(dir_param)
                pdf_value = pi * dir.pdf(data.T)
                break
        likelihood[:, k] = pdf_value

    likelihood = np.sum(likelihood, axis=1)
    likelihood = np.log(likelihood)
    likelihood = np.sum(likelihood)

    return likelihood


# plot likelihood
def plot_likelihoood(data, filepath):
    data = copy.deepcopy(data)
    data = data[2:]
    steps = len(data)

    plt.figure()
    plt.plot(range(steps), data)
    plt.savefig(filepath + 'likelihood.png')
    plt.close()


def data_scatter(data, respon, filepath):
    data = copy.deepcopy(data)
    respon = copy.deepcopy(respon)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # scatter the data colored by corresponding respons
    ax.scatter(data[:, 1], data[:, 2], c=np.array([respon[:, 0], respon[:, 1], np.zeros_like(respon[:, 1])]).T)

    plt.savefig(filepath + 'data_dist_visualization.png')


def draw_dir_contour(dist_params, data, respon, filepath):
    f = plt.figure(figsize=(8, 6))
    dist_params = copy.deepcopy(dist_params)
    plt.subplot(2, 1, 1)
    dist = Dirichlet(dist_params)
    draw_pdf_contours(dist)
    # title = r'%.3f*(%.3f, %.3f, %.3f)+%0.3f*(%.3f, %.3f, %.3f)' % tuple(alpha)
    # plt.title(title, fontdict={'fontsize': 8})
    plt.subplot(2, 1, 2)
    plot_points(data, respon)

    if not os.path.exists(filepath):
        os.mkdir(filepath)
    plt.savefig(filepath + 'dist_visualization.png')


if __name__ == '__main__':
    main()
