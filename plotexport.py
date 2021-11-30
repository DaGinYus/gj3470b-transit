"""
Created by Matthew Wong
UCSB 2021-11-30
PHYS 134L Final Project
GJ 3470b Transit
A file containing plotting functions and export
"""

import os
import logging
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

_EXPORT_DIRNAME = "output"
_DEFAULT_FIGSIZE = (9.6, 7.2)

def export_to_image(fname, fig):
    """Takes an Axes object and saves the figure. If the output
       directory doesn't already exist, creates one for the user.
    """
    try:
        os.mkdir(_EXPORT_DIRNAME)
        logging.info("Directory '%s' created", _EXPORT_DIRNAME)
    except FileExistsError:
        pass
    fpath = os.path.join(_EXPORT_DIRNAME, fname)
    fig.savefig(fpath)
    logging.info("Exported to %s", fpath)

def aper_sum_all(obj_aper_sum, aper_sum_data):
    fig, ax = plt.subplots(figsize=(_DEFAULT_FIGSIZE))
    for i, row in enumerate(aper_sum_data):
        ax.plot(row, label=f"Star {i+1}", alpha=0.6)
    ax.plot(obj_aper_sum, label="Object of Interest", color="black")
    ax.set_title(("Aperture Sum Curves for Object of Interest "
                  "and Reference Stars"))
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Normalized Flux Value")
    ax.legend(ncol=2)
    export_to_image("aper_sum_all", fig)
