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


def files_from_arg():
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
    # some image files were taken with a different filter
    EXPECTED_FILT_TYPE = "ip"
    
    fdata = []
    for fname in files_from_arg():
        with fits.open(fname) as f:
            if f[0].header["FILTER"] == EXPECTED_FILT_TYPE:
                fdata.append(f[0].data)
    print(np.array(fdata))


if __name__ == "__main__":
    transit()
