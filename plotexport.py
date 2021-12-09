"""
Created by Matthew Wong
UCSB 2021-11-30
PHYS 134L Final Project
GJ 3470b Transit
A file containing plotting and export functions
"""

import os
import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

_EXPORT_DIRNAME = "output"
_EXPORT_FORMAT = "pdf"
_DEFAULT_FIGSIZE = (9.6, 7.2)  # inches

def export_to_image(fname, fig):
    """Takes an Axes object and saves the figure. If the output
       directory doesn't already exist, creates one for the user.
    """
    try:
        os.mkdir(_EXPORT_DIRNAME)
        logging.info("Directory '%s' created", _EXPORT_DIRNAME)
    except FileExistsError:
        pass
    fpath = os.path.join(_EXPORT_DIRNAME, fname+'.'+_EXPORT_FORMAT)
    fig.savefig(fpath, format=_EXPORT_FORMAT)
    logging.info("Plot exported to %s", fpath)

def aper_sum_with_outliers(t, aper_sum_list, obj_index):
    """Plots the object flux with outliers. Each curve is normalized
       to its median.
    """
    plot_data = aper_sum_list.copy()
    obj_data = plot_data[:, 0, obj_index]
    obj_data /= np.median(obj_data)
    fig, ax = plt.subplots(figsize=(_DEFAULT_FIGSIZE))
    for i, _ in enumerate(plot_data[0, 0]):
        fluxcurve = plot_data[:, 0, i]
        fluxcurve /= np.median(fluxcurve)
        if i != obj_index:
            ax.plot(t, fluxcurve, alpha=0.6)
    ax.plot(t, obj_data, label="GJ 3470", color="black")
    ax.set_title("Aperture Sum Curves for Chosen Objects in Frame")
    ax.set_xlabel("Time Since Observation Start [h]")
    ax.set_ylabel("Normalized Flux")
    ax.legend(ncol=4)
    export_to_image("aper_sum_outliers", fig)

def aper_sum_all(t, obj_flux, ref_fluxes):
    """Plot the object flux in comparison to the flux from the
       reference stars.
    """
    fig, ax = plt.subplots(figsize=(_DEFAULT_FIGSIZE))
    for i, ref_flux in enumerate(ref_fluxes):
        ax.plot(t, ref_flux, alpha=0.6)
    ax.plot(t, obj_flux, label="GJ 3470", color="black")
    ax.set_title("Aperture Sum Curves for GJ 3470 "
                 "and Chosen Reference Stars")
    ax.set_xlabel("Time Since Observation Start [h]")
    ax.set_ylabel("Normalized Flux")
    ax.legend()
    export_to_image("aper_sum_all", fig)

def corrected_flux(t, obj_flux, obj_err):
    """Plots the object flux normalized relative to the reference
       stars.
    """
    fig, ax = plt.subplots(figsize=(_DEFAULT_FIGSIZE))
    ax.errorbar(t, obj_flux, yerr=obj_err,
                ls='', elinewidth=0.5, capsize=2,
                marker='.', color="black")
    ax.set_title("Relative Flux of GJ 3470 Corrected Using Reference Stars")
    ax.set_xlabel("Time Since Observation Start [h]")
    ax.set_ylabel("Relative Flux")
    export_to_image("corrected_flux", fig)

def fitted_flux(t, obj_flux, fit_func):
    """Scatterplots the object flux with the fit function."""
    fig, ax = plt.subplots(figsize=(_DEFAULT_FIGSIZE))
    ax.plot(t, obj_flux, ls='', marker='.', color="black")
    ax.plot(t, fit_func, color="blue")
    ax.set_title("Relative Flux of GJ 3470 Fitted Using Boxcar Function")
    ax.set_xlabel("Time Since Observation Start [h]")
    ax.set_ylabel("Relative Flux")
    export_to_image("fitted_flux", fig)
