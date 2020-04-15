========
Overview
========

PEAKO is a supervised radar Doppler spectrum peak finding algorithm. It finds the optimal parameters for detecting
peaks in cloud radar Doppler spectra using user-generated training data.

.. figure:: /example_spectrum.png
	   :width: 500 px
	   :align: center

           Cloud radar Doppler spectrum

Reference for a description and validation of the Matlab version of the algorithm: `Kalesse, Vogl, Paduraru and Luke, 2019`_

.. _Kalesse, Vogl, Paduraru and Luke, 2019: https://www.atmos-meas-tech.net/12/4591/2019/amt-12-4591-2019.html

The current release is tailored to use cloud radar Doppler spectra netcdf files. The files are in a format which is
currently under discussion in the Cloudnet community. Changes are likely to be made in the future, and Peako will have
to be adjusted to work with the most current spectra file format.
The cloudnet community will hopefully share their routines for bringing spectra files from different cloud radars into
the desired format. Ongoing discussion is happening in the `Cloudnet forum`_.

.. _Cloudnet forum: https://forum.cloudnet.fmi.fi/
