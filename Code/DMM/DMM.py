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
    True_dist_param = {
        'dir_param_1': [6, 25, 9],
        'dir_param_2': [7, 8, 23],
        'mix_coef': [0.3, 0.7]
    }
    x, label = dmm_data_generation(NO_DATA, True_dist_param)

    # DMM data save

    write_data(x, file_path=data_path + 'DMM_Samples.pkl')
    write_data(label, file_path=data_path + 'DMM_Samples_Label.pkl')

    # DMM data visualization
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
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
    likelihood = []
    for i in range(10):
        new_respons = EM_E(x, new_respons, new_dir1_param, new_dir2_param, new_Mix_coef)
        new_dir1_param, new_dir2_param, new_Mix_coef = EM_M(x, new_respons, new_dir1_param, new_dir2_param,
                                                            new_Mix_coef)
        # calc likelihood
        dists = {
            'mix_coef': new_Mix_coef,
            'dist1': dirichlet(new_dir1_param),
            'dist2': dirichlet(new_dir2_param)
        }
        likelihood.append(calc_log_likelihood(dists=dists, data=x))

        # plot the intermeidate DMM contour
        dist_param = {'dir1_params': new_dir1_param,
                      'dir2_params': new_dir2_param,

                      }
        draw_dir_contour(dist_param, data=x, respon=new_respons, filepath=results_path + 'epoch{}/'.format(i + 1))

    # ------------------------------------------------------------------------------------------------------------------
    # after EM
    # ------------------------------------------------------------------------------------------------------------------
    # save results
    est__dist_param = {'dir1_params': new_dir1_param,
                       'dir2_params': new_dir2_param,
                       'mix_coef': new_Mix_coef
                       }

    write_data(True_dist_param, file_path=results_path + 'True_dist_params.pkl')
    write_data(est__dist_param, file_path=results_path + 'Est_dist_params.pkl')
    write_data(likelihood, file_path=results_path + 'likelihood.pkl')

    # plot likelihood
    plot_likelihoood(likelihood, filepath=results_path)


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
def calc_log_likelihood(dists, data):
    dists = copy.deepcopy(dists)
    data = copy.deepcopy(data)
    mix_coef = dists.pop('mix_coef')
    no_mixtures = len(mix_coef)
    likelihood = np.zeros(shape=(data.shape[0], no_mixtures))

    for k in range(no_mixtures):
        pi = mix_coef[0]
        for key, value in dists.items():
            if str(k + 1) in key:
                pdf_value = pi * value.pdf(data.T)
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
    alphas = dist_params.values()
    for (i, alpha) in enumerate(alphas):
        plt.subplot(2, len(alphas), i + 1)
        dist = Dirichlet(alpha)
        draw_pdf_contours(dist)
        title = r'$\alpha$ = (%.3f, %.3f, %.3f)' % tuple(alpha)
        plt.title(title, fontdict={'fontsize': 8})
        plt.subplot(2, len(alphas), i + 1 + len(alphas))
        plot_points(data, respon)

    if not os.path.exists(filepath):
        os.mkdir(filepath)
    plt.savefig(filepath + 'dist_visualization.png')


if __name__ == '__main__':
    main()
