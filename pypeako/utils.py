import numpy as np
import datetime
import xarray as xr
import matplotlib
import warnings
import os

star = matplotlib.path.Path.unit_regular_star(6)
circle = matplotlib.path.Path.unit_circle()
# concatenate the circle with an internal cutout of the star
verts = np.concatenate([circle.vertices, star.vertices[::-1, ...]])
codes = np.concatenate([circle.codes, star.codes])
cut_star = matplotlib.path.Path(verts, codes)


def lin2z(array):
    """
    convert linear values to dB (for np.array or single number)
    :param array: np.array or single number
    :return:
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = 10 * np.log10(array)
        return out


def format_hms(unixtime):
    """format time stamp in seconds since 01.01.1970 00:00 UTC to HH:MM:SS
    :param unixtime: time stamp (seconds since 01.01.1970 00:00 UTC)
    """
    return datetime.datetime.utcfromtimestamp(unixtime).strftime("%H:%M:%S")


def round_to_odd(f):
    """round to odd number
    :param f: float number to be rounded to odd number
    """
    return round(f) if round(f) % 2 == 1 else round(f) + 1


def argnearest(array, value):
    """larda function to find the index of the nearest value in a sorted array, for example time or range axis

    :param array: sorted numpy array with values, list will be converted to 1D array
    :param value: value for which to find the nearest neighbor
    :return:
        index of the nearest neighbor in array
    """
    if type(array) == list:
        array = np.array(array)
    i = np.searchsorted(array, value) - 1

    if not i == array.shape[0] - 1:
        if np.abs(array[i] - value) > np.abs(array[i + 1] - value):
            i = i + 1
    return i


def mask_velocity_vectors(spec_data: list):
    """
    Mask invalid values in velocity vectors not properly masked by xarray
    :param spec_data: list of xarray DataSets containing Doppler spectra, and the variable velocity_vectors
    :return:
    """
    for i in range(len(spec_data)):
        np.putmask(spec_data[i].velocity_vectors.values, spec_data[i].velocity_vectors.values > 9000, np.nan)
    return spec_data


def mask_fill_values(spec_data: list):
    """
    Mask fill values and very small values below 1e-10 mm6 m-3 with nan
    :param spec_data: dataset containing Doppler spectra
    """

    for i in range(len(spec_data)):
        if "_FillValue" in spec_data[i].attrs:
            np.putmask(spec_data[i].doppler_spectrum.values,
                       spec_data[i].doppler_spectrum.values == spec_data._FillValue, np.nan)
        np.putmask(spec_data[i].doppler_spectrum.values,
                       spec_data[i].doppler_spectrum.values <= 1e-10, np.nan)
    return spec_data


def save_and_reload(spec_data, filenames):
    """
    Following optimization tip #3 (https://xarray.pydata.org/en/v2023.04.2/user-guide/dask.html) save intermediate
    results to disk and then load them again
    :param spec_data: list of xarray data arrays
    :param filenames: list of strings (original files that were read in)
    :return:
    """
    list_out = [f + 'temp' for f in filenames]
    for s, f in zip(spec_data, list_out):
        if not os.path.isfile(f):
            s.to_netcdf(f)
    spec_data = [xr.open_dataset(l, mask_and_scale=True, chunks={"time":10}) for l in list_out]
    #spec_data = [s.load() for s in spec_data]
    return spec_data


def get_vel_resolution(vel_bins):
    return np.nanmedian(np.diff(vel_bins))


def vel_to_ind(velocities, velbins, fill_value):
    """
    Convert velocities of found peaks to indices

    :param velocities: list of Doppler velocities
    :param velbins: Doppler velocity bins
    :param fill_value: value to be ignored in velocities list
    :return: indices of closest match for each element of velocities in velbins
    """
    indices = np.asarray([argnearest(velbins, v) if ~np.isnan(v) else fill_value for v in velocities])

    return indices


def get_chirp_offsets(specdata):
    """
    utility function to create an array of the range indices of chirp offsets, starting with [0]
    and ending with [n_range_layers]
    :param specdata: Doppler spectra DataSet containing chirp_start_indices and n_range_layers
    :return:
    """
    return np.hstack((specdata.chirp_start_indices.values, specdata.n_range_layers.values))


def find_index_in_sublist(i, training_index: list):
    """
    find the ith sample in the training_index list
    :param i: sample number
    :param training_index: list of tuples
    :return:
    """
    list_of_lengths = [len(t[0]) for t in training_index]
    b = np.cumsum(np.array(list_of_lengths))
    list_of_lengths_2 = list(range(len(list_of_lengths)))
    c = np.digitize(i, b)
    ind = np.where(np.array(list_of_lengths_2) == c)[0]
    left_bound = 0 if c == 0 else b[c-1]
    return int(ind), left_bound


def get_closest_time(time, time_array):
    """"
    :param time: datetime.datetime
    :param time_array: xr.DataArray containing time stamp
    """
    time_array = time_array.values
    if (time_array < 1e9).all() and (time_array > 3e8).all():
        time_array = (datetime.datetime(2001, 1, 1) - datetime.datetime(1970, 1, 1)).total_seconds() + time_array
    ts = (time - datetime.datetime(1970, 1, 1)).total_seconds()
    return argnearest(time_array, ts)

