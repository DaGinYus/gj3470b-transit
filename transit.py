"""
Created by Matthew Wong
UCSB 2021-11-18
PHYS 134L Final Project
GJ 3470b Transit
"""

import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


def getfiles():
    """Returns the filenames matching the command line arguments
       using the glob module.
    """
    fnames = sys.argv[1:]
    if fnames:
        return [f for f_ in [glob.glob(e) for e in fnames]
                for f in f_]
    raise SystemExit(f"Usage: {sys.argv[0]} [files]")


def transit():
    """Processes data related to exoplanet transit.
       UPDATE THIS DOCSTRING LATER
    """
    fdata = [fits.open(f) for f in getfiles()]


if __name__ == "__main__":
    transit()
