"""
Created by Matthew Wong
UCSB 2021-11-18
PHYS 134L Final Project
GJ 3470b Transit
This is the main file handling the data reduction
"""

import sys
import os
import logging
import plotexport
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
_EXCLUDE_OUTLIERS = (0, 1, 2, 3, 6, 15)

def enable_logging():
    """Configures logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()]
    )

def files_from_arg():
    """Checks if there are filenames passed in from arguments and then
       passes them to the program.
    """
    fnames = sys.argv[1:]
    if not fnames:
        raise SystemExit(f"Usage: {sys.argv[0]} [files]")

    return fnames

def match_star(sources, match_x, match_y, threshold=5):
    """Returns the index of the star matching the coordinates most closely.
       Takes a list of sources from DAOStarFinder as input.
    """
    source_x = sources["xcentroid"].data
    source_y = sources["ycentroid"].data
    dist = np.sqrt((source_x - match_x)**2 + (source_y - match_y)**2)
    found_sources = np.where(dist < threshold)[0]
    if found_sources:
        return found_sources[0]
    return None

def create_errormap(frame, header):
    """Creates an error map based on the gain value in the header and
       the provided image frame.
    """
    gain = header["GAIN"]
    return np.sqrt(np.abs(frame)/gain)

def extract_photometry(positions, frame, errormap):
    """Perfoms aperture photometry with sigma clipping on the provided
       frame. 
    """
    aperture = CircularAperture(positions, r=_APERTURE_RADIUS)
    annulus_aperture = CircularAnnulus(positions, r_in=_ANNULUS_RADIUS_IN,
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

def load_data():
    """Returns two arrays, one storing the image data for each FITS file,
       and another storing the header data."""
    fheaders = []
    fdata = []
    for fname in files_from_arg():
        with fits.open(fname, memmap=False) as f:
            # some image files were taken with a different filter
            if f[0].header["FILTER"] == _EXPECTED_FILT_TYPE:
                fheaders.append(f[0].header)
                fdata.append(f[0].data)
    logging.info("%i files read", len(fdata))
    return fdata, fheaders

def find_ref_stars(frame):
    """Returns the positions of the 20 brightest stars as reference stars
       for photometry. Note that the object of interest is included in this list.
    """
    _, bg_median, bg_std = sigma_clipped_stats(frame, sigma=3.0)
    daofind = DAOStarFinder(fwhm=6.0, threshold=10.0*bg_std)
    sources = daofind(frame - bg_median)

    # take the 20 brightest points
    sort_descending_value = np.argsort(sources["peak"])[::-1]
    ref_star_indices = sort_descending_value[:20]
    ref_stars = sources[ref_star_indices]
    logging.info("Reference stars found")
    return ref_stars

def gen_aper_sum_list(data, headers, initial_pos, initial_ref_pos):
    """Generates a list of background subtracted aperture sum values per star,
       per frame. The output is an NxN numpy array.
    """
    aper_list = []
    for i, frame in enumerate(data):
        header = headers[i]
        ref_pos = np.array([header["CRPIX1"], header["CRPIX2"]])
        offset = ref_pos - initial_ref_pos
        positions = initial_pos + offset
        errormap = create_errormap(frame, header)
        phot = extract_photometry(positions, frame, errormap)
        aper_list.append(phot["aper_sum_bkgsub"].value)
        for col in phot.colnames:
            phot[col].info.format = '%.8g'  # for consistent table output
    logging.info("Extracted photometry for %i stars", len(aper_list[0]))
    return np.array(aper_list)

def clean_aper_data(aper_sum_data, obj_index):
    """Takes as input an NxN np array where each row corresponds to the
       (background subtracted) aperture sum data for each frame. This function
       removes the outliers specified in _EXCLUDE_OUTLIERS
       and returns two arrays:

          - The first array returned is the photometry for the 
            object of interest
          - The second array is an NxN array pruned of the outliers.
            Note that the dimensions are now flipped. Each row now corresponds
            to data for a specific star, not for a specific frame.
    """
    cleaned_aper_sum = []
    for i in range(len(aper_sum_data[0])):
        aper_sum_data[:, i] /= np.median(aper_sum_data[:, i])  # norm the data
        if i not in _EXCLUDE_OUTLIERS and i != obj_index:
            cleaned_aper_sum.append(aper_sum_data[:, i])
    obj_data = aper_sum_data[:, obj_index]
    return obj_data, np.array(cleaned_aper_sum)


def transit():
    """Processes data related to exoplanet transit.
       UPDATE THIS DOCSTRING LATER
    """
    enable_logging()
    fdata, fheaders = load_data()

    # find stars based on first frame
    ref_stars = find_ref_stars(fdata[0])

    # the index of the object of interest
    obj_index = match_star(ref_stars, _OBJ_XCOORD, _OBJ_YCOORD)
    if not obj_index:
        raise SystemExit(f"No object found at {_OBJ_XCOORD}, {_OBJ_YCOORD}")

    initial_pos = np.transpose((ref_stars["xcentroid"],
                                ref_stars["ycentroid"]))
    initial_ref_pos = np.array([fheaders[0]["CRPIX1"],
                                fheaders[0]["CRPIX2"]])

    aper_sum_list = gen_aper_sum_list(fdata, fheaders, initial_pos,
                                      initial_ref_pos)

    obj_aper_sum, aper_sum_data = clean_aper_data(aper_sum_list, obj_index)
    plotexport.aper_sum_all(obj_aper_sum, aper_sum_data)


if __name__ == "__main__":
    transit()
