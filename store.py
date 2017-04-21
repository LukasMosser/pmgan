import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

import pandas as pd
from scipy.optimize import curve_fit

import json
import os


def correlation_function(cov):
    p = cov[0]
    return (cov-p**2)/(p-p**2)

def straight_line_at_origin(porosity):
    def func(x, a):
        return a * x + porosity
    return func


def to_json(porosities):
    pass


def plot_sections():
    pass


def radial_average(cov):
    avg = np.mean(cov, axis=0)
    return avg


def import_s2_orig(filename):
    orig_cov = pd.read_csv(filename)

    radial_averages_orig = np.mean(orig_cov.values.T, axis=0)

    directional_averages_orig = orig_cov.values.T

    return radial_averages_orig, directional_averages_orig


def process_s2_data(cov):
    cov = np.array([[c['x'], c['y'], c['z']] for c in cov])

    directional_averages = np.mean(cov, axis=0)

    radial_averages = np.mean(cov.reshape(-1, cov.shape[-1]), axis=0)
    radial_std = np.std(cov.reshape(-1, cov.shape[-1]), axis=0)

    directional_std = np.std(cov, axis=0)

    return radial_averages, radial_std, directional_averages, directional_std


def plot_s2(dir, filename, cov, logging):
    logging.debug(filename)
    radial_averages_orig, directional_averages_orig = import_s2_orig(filename)
    logging.debug('got orig')
    radial_averages, radial_std, directional_averages, directional_std = process_s2_data(cov)
    logging.debug('got s2')
    plot_radial_averaged_s2(dir, radial_averages, radial_std, radial_averages_orig, logging)
    logging.debug('plot directional')
    plot_directional_s2(dir, directional_averages, directional_std, directional_averages_orig)
    logging.debug('plot directional')

def plot_radial_averaged_s2(dir, radial_averages, radial_std, radial_averages_orig, logging):
    plt.ioff()
    logging.debug('turned off')
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    porosity_avg = radial_averages[0]
    porosity_avg_orig = radial_averages_orig[0]

    ax.errorbar(range(len(radial_averages)), radial_averages, yerr=radial_std, c="black", elinewidth=1,
                fmt='-', label=r"$Synthetic$", linewidth=3)
    ax.plot(range(len(radial_averages_orig)), radial_averages_orig, linestyle="--", linewidth=4, c="red", label=r"$Original$")

    ax.plot([0, 20], [porosity_avg, porosity_avg], linestyle="--", color="black", linewidth=3)

    ax.plot([0, 20], [porosity_avg_orig, porosity_avg_orig], linestyle="--", color="red", linewidth=3)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)

    ax.set_xlabel(r"$Lag \ Distance \ r \ [voxels]$", fontsize=36)

    ax.set_ylabel(r"$S_2(r)$", fontsize=36)
    ax.set_xlim(-1, 150)

    ax.grid()
    ax.legend(fontsize=32)
    logging.debug('plotted')
    fig.savefig(os.path.join(dir, "radial_averaged_s2.png"), bbox_extra_artists=None, bbox_inches='tight', dpi=72)
    logging.debug('saved')

def plot_directional_s2(dir, directional_averages, directional_std, directional_averages_orig):
    plt.ioff()
    fig, ax = plt.subplots(1, 3, figsize=(36, 12))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.05)
    for i, (j, direc) in zip(range(0, 6, 2), enumerate([r"$x$", r"$y$", r"$z$"])):
        ax[j].errorbar(range(len(directional_averages[j])), directional_averages[j],
                       yerr=directional_std[j], c="black", fmt='-')
        ax[j].plot(range(len(directional_averages_orig[j])), directional_averages_orig[j], linestyle="--", linewidth=4, c="red")

        for tick in ax[j].xaxis.get_major_ticks():
            tick.label.set_fontsize(20)

        for tick in ax[j].yaxis.get_major_ticks():
            tick.label.set_fontsize(20)

    for j, direc in enumerate([r"$x$", r"$y$", r"$z$"]):
        ax[j].set_title(direc + r"$-Direction$", fontsize=36, y=1.02)
        ax[j].set_xlabel(r"$Lag \ Distance \ r($" + direc + "$) \ [voxels]$", fontsize=36)

    ax[0].set_ylabel(r"$S_2(r)$", fontsize=36)
    for ax_handle in ax.flatten():
        ax_handle.set_xlim(-1, 150)
        ax_handle.grid()

    fig.savefig(os.path.join(dir, "directional_s2.png"), bbox_extra_artists=None, bbox_inches='tight', dpi=72)


def plot_images(dir, imgs, z=100):
    plt.ioff()
    fig, ax = plt.subplots(2, 5, figsize=(10, 10))

    for i in range(0, 5):
        img = ax[0, i].imshow(imgs[i][:, :, z], interpolation='nearest', cmap="Greys")

    for i in range(5, 10):
        img = ax[1, i - 5].imshow(imgs[i][:, :, z], interpolation='nearest', cmap="Greys")

    for a in ax.flatten():
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.set_aspect('equal')
        a.set_xticks([])
        a.set_yticks([])

    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    plt.axis('off')

    fig.savefig(os.path.join(dir, "berea_comparison.png"), bbox_inches='tight', dpi=300)