import numpy as np
import datetime


def lin2z(array):
    """
    convert linear values to dB (for np.array or single number)
    :param array: np.array or single number
    :return:
    """
    return 10 * np.ma.log10(array)


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


def mask_velocity_vectors(spec_data):
    for i in range(len(spec_data)):
        np.putmask(spec_data[i].velocity_vectors.values, spec_data[i].velocity_vectors.values>9000, np.nan)
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