"""
Created by Matthew Wong
UCSB 2021-11-18
PHYS 134L Final Project
GJ 3470b Transit
"""

import contextlib
import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.aperture import (
    aperture_photometry,
    CircularAperture,
    CircularAnnulus)

_EXPECTED_FILT_TYPE = "ip"
_APERTURE_RADIUS = 15
_ANNULUS_RADIUS_IN = 17
_ANNULUS_RADIUS_OUT = 25
_OBJ_XCOORD = 1006.22
_OBJ_YCOORD = 1046.99

def files_from_arg():
    """Checks if there are filenames passed in from arguments and then
       passes them to the program.
    """
    fnames = sys.argv[1:]
    if not fnames:
        raise SystemExit(f"Usage: {sys.argv[0]} [files]")

    return fnames

def match_star(daofind_list, match_x, match_y, threshold=5):
    """Returns the index of the star matching the coordinates most closely.
       Takes a list of sources from DAOStarFinder as input.
    """
    source_x = daofind_list["xcentroid"].data
    source_y = daofind_list["ycentroid"].data
    dist = np.sqrt((source_x - match_x)**2 + (source_y - match_y)**2)
    found_sources = np.where(dist < threshold)[0]
    if found_sources:
        return found_sources[0]

def create_errormap(frame, header):
    """Creates an error map based on the gain value in the header and 
       the provided image frame.
    """
    gain = header["GAIN"]
    return np.sqrt(frame/gain)

def extract_photometry(pos, frame, errormap):
    """Perfoms aperture photometry with sigma clipping on the provided
       frame.
    """
    aperture = CircularAperture(pos, r=_APERTURE_RADIUS)
    annulus_aperture = CircularAnnulus(pos, r_in=_ANNULUS_RADIUS_IN,
                                       r_out=_ANNULUS_RADIUS_OUT)
    annulus_masks = annulus_aperture.to_mask(method="center")

    bkg_median = []
    for mask in annulus_masks:
        annulus_data = mask.multiply(frame)
        annulus_data_1d = annulus_data[mask.data > 0]
        _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
        bkg_median.append(median_sigclip)
    bkg_median = np.array(bkg_median)
    phot = aperture_photometry(frame, aperture, error=errormap)
    phot["annulus_median"] = bkg_median
    phot["aper_bkg"] = bkg_median * aperture.area
    phot["aper_sum_bkgsub"] = phot["aperture_sum"] - phot["aper_bkg"]
    return phot

def transit():
    """Processes data related to exoplanet transit.
       UPDATE THIS DOCSTRING LATER
    """
    fheaders = []
    fdata = []
    for fname in files_from_arg():
        with fits.open(fname, memmap=False) as f:
            # some image files were taken with a different filter
            if f[0].header["FILTER"] == _EXPECTED_FILT_TYPE:
                fheaders.append(f[0].header)
                fdata.append(f[0].data)

    _, bg_median, bg_std = sigma_clipped_stats(fdata[0], sigma=3.0)
    daofind = DAOStarFinder(fwhm=6.0, threshold=10.0*bg_std)
    sources = daofind(fdata[0] - bg_median)
    positions = np.transpose((sources["xcentroid"], sources["ycentroid"]))

    # take the 20 brightest points as reference stars
    sort_descending_value = np.argsort(sources["peak"])[::-1]
    ref_star_indices = sort_descending_value[:20]
    ref_star_pos = positions[ref_star_indices]

    # the index of the object of interest
    obj_index = match_star(sources[ref_star_indices],
                           _OBJ_XCOORD, _OBJ_YCOORD)
    initial_x_ref = fheaders[0]["CRPIX1"]
    initial_y_ref = fheaders[0]["CRPIX2"]

    for i, frame in enumerate(fdata):
        header = fheaders[i]
        errormap = create_errormap(frame, header)
        phot = extract_photometry(ref_star_pos, frame, errormap)
    for col in phot.colnames:
        phot[col].info.format = '%.8g'  # for consistent table output
    print(phot)


if __name__ == "__main__":
    transit()
