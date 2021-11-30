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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.aperture import (
    aperture_photometry,
    CircularAperture,
    CircularAnnulus
)

import plotexport

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

def load_data(fname):
    """Reads the data in a FITS file and returns the header and 
       image data.
    """
    fheader = None
    fdata = None
    with fits.open(fname, memmap=False) as f:
        # some image files were taken with a different filter
        if f[0].header["FILTER"] == _EXPECTED_FILT_TYPE:
            fheader = f[0].header
            fdata = f[0].data
    return fdata, fheader

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

def get_aper_sum(frame, header, initial_pos, initial_ref_pos):
    """Returns the (background subtracted) aperture sum for the reference
       stars within a frame, along with the error.
    """
    ref_pos = np.array([header["CRPIX1"], header["CRPIX2"]])
    offset = ref_pos - initial_ref_pos
    positions = initial_pos + offset
    errormap = create_errormap(frame, header)
    phot = extract_photometry(positions, frame, errormap)
    return [phot["aper_sum_bkgsub"].value,
            phot["aperture_sum_err"].value]

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
    cleaned_aper_err = []
    aper_sum = aper_sum_data[:, 0]
    aper_err = aper_sum_data[:, 1]
    for i in range(len(aper_sum[0])):
        # norm the data
        aper_median = np.median(aper_sum[:, i])
        aper_sum[:, i] /= aper_median
        aper_err[:, i] /= aper_median
        if i not in _EXCLUDE_OUTLIERS and i != obj_index:
            cleaned_aper_sum.append(aper_sum[:, i])
            cleaned_aper_err.append(aper_err[:, i])
    obj_data = aper_sum[:, obj_index]
    obj_err = aper_err[:, obj_index]
    return (obj_data, obj_err,
            np.array(cleaned_aper_sum),
            np.array(cleaned_aper_err))

def initial_setup(fdata, fheader):
    """Sets up some basic variables based on data in the first frame.
         - obj_index is the index that the object occurs at in the dataset.
         - initial_pos is an array containing the inital positions of the
           reference stars.
         - initial_ref_pos is the location of the reference pixel defined
           in the header of the FITS file.
    """
    ref_stars = find_ref_stars(fdata)
    obj_index = match_star(ref_stars, _OBJ_XCOORD, _OBJ_YCOORD)
    if not obj_index:
        raise SystemExit(f"No object found at "
                         f"{_OBJ_XCOORD}, {_OBJ_YCOORD}")
    initial_pos = np.transpose((ref_stars["xcentroid"],
                                ref_stars["ycentroid"]))
    initial_ref_pos = np.array([fheader["CRPIX1"],
                                fheader["CRPIX2"]])
    return obj_index, initial_pos, initial_ref_pos

def normalize_flux(obj_flux, obj_err, ref_fluxes):
    """Normalize the object flux, to account for atmospheric effects.
       At each frame, divide the object flux by the median of all the 
       other fluxes. Returns a normalized 1d array of flux values.
    """
    median = np.median(ref_fluxes, axis=0)
    return (obj_flux / median, obj_err / median)

def transit():
    """Processes data related to exoplanet transit.
       UPDATE THIS DOCSTRING LATER
    """
    enable_logging()

    aper_sum_list = []
    file_count = 0
    for fname in files_from_arg():
        fdata, fheader = load_data(fname)
        if fdata is not None:
            # find stars based on first frame
            if file_count == 0:
                obj_index, initial_pos, \
                    initial_ref_pos = initial_setup(fdata, fheader)

            aper_sum_list.append(get_aper_sum(fdata, fheader, initial_pos,
                                              initial_ref_pos))
            file_count += 1
    logging.info("Aperture sums extracted from %i files", file_count)

    aper_sum_list = np.array(aper_sum_list)
    obj_flux, obj_err, ref_fluxes, ref_err \
        = clean_aper_data(aper_sum_list, obj_index)
    plotexport.aper_sum_all(obj_flux, ref_fluxes)

    norm_obj_flux, norm_obj_err = normalize_flux(obj_flux, obj_err,
                                                 ref_fluxes)
    plotexport.corrected_flux(norm_obj_flux, norm_obj_err)


if __name__ == "__main__":
    transit()
