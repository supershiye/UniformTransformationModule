import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import torchvision.transforms as transforms
from torchvision.utils import make_grid

from dSpirtes_Dataset import dSprites_Dataset as dSpritesDataset

import numpy as np
import os

import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from seaborn import heatmap, histplot


# from sklearn.neighbors import KernelDensity

from scipy.stats import gaussian_kde
from scipy.signal import argrelextrema
from scipy.stats import norm, gaussian_kde

# Univariate clustering module
def UnivariateClustering(data, merge_threshold=1e-3):
    data = np.asarray(data).flatten()
    kernel = gaussian_kde(data)
    s = np.linspace(-6, 6, 1000)
    e = kernel(s)
    minimas = s[argrelextrema(e, np.less)[0]]
    maximas = s[argrelextrema(e, np.greater)[0]]

    if len(maximas) <= 1:
        return np.zeros(len(data)), maximas, minimas, np.array([np.var(data)]), np.array([1.0])

    labels = np.searchsorted(minimas, data, side='right')
    
    weights = np.array([np.sum(labels == i) for i in range(len(maximas))]) / len(data)

    # Merging clusters
    merged_indices = []
    for i, weight in enumerate(weights):
        if weight < merge_threshold:
            # Find the closest maxima with higher weight
            diff = np.abs(maximas - maximas[i])
            diff[i] = np.inf  # Ignore the current maxima
            closest_idx = np.argmin(diff)
            labels[labels == i] = closest_idx
            merged_indices.append(i)

    # Update maximas, minimas, variances, and weights after merging
    merged_indices = set(merged_indices)
    maximas = np.array([m for i, m in enumerate(maximas) if i not in merged_indices])
    minimas = np.array([m for i, m in enumerate(minimas) if i not in merged_indices])
    labels = np.searchsorted(minimas, data, side='right')
    variances = np.array([np.var(data[labels == i]) for i in range(len(maximas))])
    weights = np.array([np.sum(labels == i) for i in range(len(maximas))]) / len(data)

    return labels, maximas, minimas, variances, weights


def apply_clustering_to_matrix(matrix):
    results = []
    for column in matrix.T:
        labels, maximas, minimas, variances, weights = UnivariateClustering(column)
        result = {
            'labels': labels,
            'maximas': maximas,
            'minimas': minimas,
            'variances': variances,
            'weights': weights
        }
        results.append(result)
    return results


class GaussianMixtureCDF(nn.Module):
    def __init__(self, means, variances, weights, use_sigmoid = 0):
        super(GaussianMixtureCDF, self).__init__()

        self.means = nn.Parameter(torch.tensor(means)).requires_grad_(False)
        self.variances = nn.Parameter(torch.tensor(variances)).requires_grad_(False)
        self.weights = nn.Parameter(torch.tensor(weights,dtype = torch.float32)).requires_grad_(False)
        self.use_sigmoid = use_sigmoid  

    def forward(self, x):
        cdf = torch.zeros_like(x)
        x = torch.sigmoid(x) if self.use_sigmoid else x
        if len(self.means) == 1:
            gaussian = torch.distributions.Normal(loc=self.means[0], scale=self.variances[0].sqrt())
            cdf = gaussian.cdf(x)
        else:
            
            for mean, variance, weight in zip(self.means, self.variances, self.weights):
                # print(mean, variance, weight)
                # print(mean.shape, variance.shape, weight.shape)
                gaussian = torch.distributions.Normal(loc=mean, scale=variance.sqrt())
                cdf += weight * gaussian.cdf(x)

        return cdf*8-4 # scale the output to [-4,4]
    

class UniformTransformer(nn.Module):
    def __init__(self, clusters):
        super(UniformTransformer, self).__init__()
        self.modules = []
        self.clusters = clusters
        for cluster in clusters:
            self.modules.append(GaussianMixtureCDF(cluster['maximas'], cluster['variances'], cluster['weights']))

    def forward(self, x):
        z_gaussian_cdf = torch.zeros_like(x)
        for i, module in enumerate(self.modules):
            z_gaussian_cdf[:, i] = module(x[:, i])

        return z_gaussian_cdf