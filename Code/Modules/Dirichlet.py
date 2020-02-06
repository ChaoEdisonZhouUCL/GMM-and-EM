'''Functions for drawing contours of Dirichlet distributions.'''

# Author: Thomas Boggs

from functools import reduce

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

_corners = np.array([[0, 0], [1, 0], [0.5, 0.75 ** 0.5]])
_triangle = tri.Triangulation(_corners[:, 0], _corners[:, 1])
_midpoints = [(_corners[(i + 1) % 3] + _corners[(i + 2) % 3]) / 2.0 \
              for i in range(3)]


def xy2bc(xy, tol=0.0):
    '''Converts 2D Cartesian coordinates to barycentric.
    Arguments:
        `xy`: A length-2 sequence containing the x and y value.
    '''
    s = [(_corners[i] - _midpoints[i]).dot(xy - _midpoints[i]) / 0.75 \
         for i in range(3)]
    return np.clip(s, tol, 1.0 - tol)


class Dirichlet(object):
    def __init__(self, dist_params):
        '''
        Creates Dirichlet distribution with parameter `dist_params`.
        e.x.,dist_params={
        'mix_coef':[0.5,0.5],
        'dir1_params':[2,3,19],
        'dir2_params':[17,10,7]
        }
        '''
        from scipy.stats import dirichlet
        self.mix_coef = dist_params.pop('mix_coef')
        self.K = len(self.mix_coef)
        self.dir_mixtures = [dirichlet(dir_param) for dir_param in dist_params.values()]

    def pdf(self, x):
        '''Returns pdf value for `x`.'''
        pdf = 0
        for k in range(self.K):
            pdf += self.mix_coef[k] * self.dir_mixtures[k].pdf(x)
        return pdf


def draw_pdf_contours(dist, border=False, nlevels=200, subdiv=8, **kwargs):
    '''Draws pdf contours over an equilateral triangle (2-simplex).
    Arguments:
        `dist`: A distribution instance with a `pdf` method.
        `border` (bool): If True, the simplex border is drawn.
        `nlevels` (int): Number of contours to draw.
        `subdiv` (int): Number of recursive mesh subdivisions to create.
        kwargs: Keyword args passed on to `plt.triplot`.
    '''

    refiner = tri.UniformTriRefiner(_triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)
    pvals = [dist.pdf(xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)]
    # pvals = []
    # for xy in zip(trimesh.x, trimesh.y):
    #     pdf_value = dist.pdf(xy2bc(xy))
    #     pvals.append(pdf_value)

    plt.tricontourf(trimesh, pvals, nlevels, **kwargs)
    plt.axis('equal')
    plt.xlim(0, 1)
    plt.ylim(0, 0.75 ** 0.5)
    plt.axis('off')
    if border is True:
        plt.hold(1)
        plt.triplot(_triangle, linewidth=1)


def plot_points(X, respon, barycentric=True, border=True, **kwargs):
    '''Plots a set of points in the simplex.
    Arguments:
        `X` (ndarray): A 2xN array (if in Cartesian coords) or 3xN array
                       (if in barycentric coords) of points to plot.
        `barycentric` (bool): Indicates if `X` is in barycentric coords.
        `border` (bool): If True, the simplex border is drawn.
        kwargs: Keyword args passed on to `plt.plot`.
    '''
    if barycentric is True:
        X = X.dot(_corners)
    plt.scatter(X[:, 0], X[:, 1], c=np.array([respon[:, 0], respon[:, 1], np.zeros_like(respon[:, 1])]).T)
    plt.axis('equal')
    plt.xlim(0, 1)
    plt.ylim(0, 0.75 ** 0.5)
    plt.axis('off')
    if border is True:
        plt.triplot(_triangle, linewidth=1)
