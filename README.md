# GJ3470b Transit
Final project analyzing the transit of exoplanet Gliese 3470b for PHYS 134L taught at the University of California, Santa Barbara.
The data reduction follows the process outlined in the documentation of `photutils` for `DAOStarFinder` and aperture photometry using an annulus. 
A simple boxcar function was fitted to the resulting light curve for analysis.

The input data used is data from LCOGT McDonald Observatory, taken on 2013-03-19. The data is used in the following publication: https://arxiv.org/abs/1406.6437. My program performed basic data reduction, reading in files in batches to avoid memory issues. I wrote it with optimization in mind. On a 2017 i7 MacBook Pro its runtime is just under a minute. Not bad for processing nearly 400 files. The output files of my program are viewable in the output folder.
