import xarray as xr
import scipy
import numpy as np
import datetime
import math
import warnings
import scipy.signal as si
import copy
import os
from scipy.optimize import differential_evolution
import random
import matplotlib
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial

# TODO specify directory for plots and write some pdfs out with nice naming


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
    return int(np.ceil(f / 2.) * 2 + 1)


def argnearest(array, value):
    """larda function to find the index of the nearest value in a sorted array
    for example time or range axis

    :param array: sorted numpy array with values, list will be converted to 1D array
    :param value: value to find
    :return:
        index
    """
    if type(array) == list:
        array = np.array(array)
    i = np.searchsorted(array, value) - 1

    if not i == array.shape[0] - 1:
        if np.abs(array[i] - value) > np.abs(array[i + 1] - value):
            i = i + 1
    return i


def peak_width(spectrum, pks, left_edge, right_edge, rel_height=0.5):
    """
    Calculates the width (at half height) of each peak in a signal. Returns

    :param spectrum: 1-D ndarray, input signal
    :param pks: 1-D ndarray, indices of the peak locations
    :param left_edge: 1-D ndarray, indices of the left edges of each peak
    :param right_edge: 1-D ndarray, indices of the right edges of each peak
    :param rel_height: float, at which relative height compared to the peak height the width should be computed.
    Default is 0.5, i.e. the peak width at half-height is computed.
    :return: width: array containing the width in # of Doppler bins

    """
    # initialize empty lists
    left_ps = []
    right_ps = []
    # calculate the reference height for each peak, i.e. half height if rel_height is set to 0.5
    try:
        ref_height = spectrum[left_edge] + (spectrum[pks] - spectrum[left_edge]) * rel_height
    except IndexError:
        raise IndexError('Likely there is an index out of bounds or empty. left edge, right edge, pks:',
                         left_edge, right_edge, pks)
    # loop over all peaks
    for i in range(len(pks)):
        # if y-value of the left peak edge is greater than the reference height, left edge is used as left position
        if spectrum[left_edge[i]] >= ref_height[i]:
            left_ps.append(left_edge[i])
        # else the maximum index in the interval from left edge to peak with y-value smaller/equal to the
        # reference height is used
        else:
            left_ps.append(max(np.where(spectrum[left_edge[i]:pks[i]] <= ref_height[i])[0]) + left_edge[i])
        # if y-value of the right peak edge is greater than the reference height, right edge is used as right
        # position
        if spectrum[right_edge[i]] >= ref_height[i]:
            right_ps.append(right_edge[i])
        # else the minimum index in the interval from peak to right edge smaller/equal the reference height is used
        else:
            # same as with left edge but in other direction; the index of the peak has to be added
            right_ps.append(min(np.where(spectrum[pks[i]:right_edge[i] + 1] <= ref_height[i])[0]) + pks[i])

    width = [j - i for i, j in zip(left_ps, right_ps)]
    # calculate width in relation to the indices of the left and right position (edge)
    return np.asarray(width)


def find_edges(spectrum, fill_value, peak_locations):
    """
    Find the indices of left and right edges of peaks in a spectrum

    :param spectrum: a single spectrum in linear units
    :param peak_locations: indices of peaks detected for this spectrum
    :param fill_value: The fill value which indicates the spectrum is below noise floor
    :return: left_edges: list of indices of left edges,
             right_edges: list of indices of right edges
    """
    left_edges = []
    right_edges = []

    for p_ind in range(len(peak_locations)):
        # start with the left edge
        p_l = peak_locations[p_ind]

        # set first estimate of left edge to last bin before the peak
        closest_below_noise_left = np.where(spectrum[0:p_l] == fill_value)
        if len(closest_below_noise_left[0]) == 0:
            closest_below_noise_left = 0
        else:
            # add 1 to get the first bin of the peak which is not fill_value
            closest_below_noise_left = max(closest_below_noise_left[0]) + 1

        if p_ind == 0:
            # if this is the first peak, the left edge is the closest_below_noise_left
            left_edge = closest_below_noise_left
        elif peak_locations[p_ind - 1] > closest_below_noise_left:
            # merged peaks
            left_edge = np.argmin(spectrum[peak_locations[p_ind - 1]: p_l])
            left_edge = left_edge + peak_locations[p_ind - 1]
        else:
            left_edge = closest_below_noise_left

        # Repeat for right edge
        closest_below_noise_right = np.where(spectrum[p_l:-1] == fill_value)
        if len(closest_below_noise_right[0]) == 0:
            # if spectrum does not go below noise (fill value), set it to the last bin
            closest_below_noise_right = len(spectrum) - 1
        else:
            # subtract one to obtain the last index of the peak
            closest_below_noise_right = min(closest_below_noise_right[0]) + p_l - 1

        # if this is the last (rightmost) peak, this first guess is the right edge
        if p_ind == (len(peak_locations) - 1):
            right_edge = closest_below_noise_right

        elif peak_locations[p_ind + 1] < closest_below_noise_right:
            right_edge = np.argmin(spectrum[p_l:peak_locations[p_ind + 1]]) + p_l
        else:
            right_edge = closest_below_noise_right

        left_edges.append(np.int(left_edge))
        right_edges.append(np.int(right_edge))

    return left_edges, right_edges


def area_above_floor(left_edge, right_edge, spectrum, noise_floor, velbins):
    """
    return the area below the spectrum between left and right edge (Riemann sum approximation of the area)

    :param left_edge: index (x value) of left edge from which to start integrating (or summing up).
    :param right_edge: index (x value) of right edge up to where the spectrum is integrated (or summed up).
    :param spectrum: the y values below which the area is approximated.
    :param noise_floor: the (constant) y value above which the area is approximated.
    :param velbins: Doppler velocity bins, the scaling of the x axis.
    :return: area
    """
    spectrum_above_noise = [max(x, 0) for x in list(spectrum-noise_floor)]
    spectrum_above_noise = np.array(spectrum_above_noise)
    # Riemann sum (approximation of area):
    area = np.nansum(spectrum_above_noise[left_edge:right_edge]) * (velbins[1]-velbins[0])

    return area


def overlapping_area(edge_list_1, edge_list_2, spectrum, noise_floor, velbins):
    """
    Compute maximum overlapping area of hand-marked peaks and algorithm-detected peaks in a radar Doppler spectrum
    :param edge_list_1: indices of peak edges of either user marked peaks or algorithm found peaks
    :param edge_list_2: indices of peak edges of the other peaks
    :param spectrum: ndarray containing reflectivity in dB units, contains nan values
    :param noise_floor: value of noise floor
    :param velbins: ndarray of same length as spectrum, from -Nyquist to +Nyquist Doppler velocity (m/s)
    """
    max_area = 0
    peak_ind_1 = None
    peak_ind_2 = None

    for i1 in range(int(len(edge_list_1[0]))):
        for i2 in range(int(len(edge_list_2[0]))):
            this_area = compute_overlapping_area(i1, i2, edge_list_1, edge_list_2, spectrum, noise_floor, velbins)
            if this_area > max_area:
                peak_ind_1 = i1
                peak_ind_2 = i2
                max_area = this_area
    return peak_ind_1, peak_ind_2, max_area


def compute_overlapping_area(i1, i2, edge_list_1, edge_list_2, spectrum, noise_floor, velbins):
    """ Compute overlapping area of two peaks defined by their edge indices in a radar Doppler spectrum

            :param i1: index one
            :param i2: index two
            :param edge_list_1: list of two lists containing left and right edges of detected peaks method (1)
            :param edge_list_2: list of two lists containing left and right edges of detected peaks method (2)
            :param spectrum: cloud radar Doppler spectrum (y values)
            :param noise_floor: minimum value (y value) of the spectrum above which the area will be approximated
            :param velbins: cloud radar Doppler bins (x values)

        """
    #print(i1, i2, edge_list_1,edge_list_2,spectrum,noise_floor,velbins)
    left_edge_overlap = max(edge_list_1[0][i1], edge_list_2[0][i2])
    leftest_edge = min(edge_list_1[0][i1], edge_list_2[0][i2])
    right_edge_overlap = min(edge_list_1[1][i1], edge_list_2[1][i2])
    rightest_edge = max(edge_list_1[1][i1], edge_list_2[1][i2])

    # Compute edges of joint area and of region outside joint area
    area = area_above_floor(left_edge_overlap, right_edge_overlap, spectrum, noise_floor, velbins)
    if area > 0:
        area = area - area_above_floor(leftest_edge, left_edge_overlap, spectrum, noise_floor, velbins)
        area = area - area_above_floor(right_edge_overlap, rightest_edge, spectrum, noise_floor, velbins)

    return area


def vel_to_ind(velocities, velbins, fill_value):
    """
    Convert velocities of found peaks to indices

    :param velocities: list of Doppler velocities
    :param velbins: Doppler velocity bins
    :param fill_value: value to be ignored in velocities list
    :return: indices of closest match for each element of velocities in velbins
    """

    indices = np.asarray([argnearest(velbins, v) for v in velocities])
    indices[velocities == fill_value] = fill_value

    return indices


def plot_timeheight_numpeaks(data, maxpeaks=5, key='peaks'):
    """

    :param data: xarray.dataset containing range, time and number of peaks
    :param maxpeaks: maximum number of peaks
    :param key: key (name) of the number of peaks in data
    :return: fig, ax matplotlib.pyplot.subplots()
    """

    chirp = len(data.chirp)
    fig, ax = plt.subplots(1, figsize=[10, 5.7])
    dt_list = [datetime.datetime.utcfromtimestamp(time) for time in data.time.values]
    cmap = plt.cm.viridis  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]


    # create the new map
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap.N)
    # define the bins and normalize
    bounds = np.linspace(0, maxpeaks, maxpeaks+1)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    for c in range(chirp):
        pcmesh = ax.pcolormesh(matplotlib.dates.date2num(dt_list[:]),
                               data[f'C{c+1}range'].values/1000, np.transpose(np.sum(data[f'C{c+1}{key}'].values > -900,
                                                                        axis=2)), cmap=cmap, vmin=0, norm=norm,
                               vmax=maxpeaks)

    bar = fig.colorbar(pcmesh)
    ax.set_xlabel("Time [UTC]", fontweight='semibold', fontsize=12)
    ax.set_ylabel("Range [km]", fontweight='semibold', fontsize=12)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 5)))
    ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 1)))
    fig.tight_layout()
    bar.ax.set_ylabel('number of peaks', fontweight='semibold', fontsize=12)
    return fig, ax


def plot_spectrum_peako_peaks(peaks_dataset, spec_dataset, height, time, key='PeakoPeaks'):

    ranges = np.array([j for i in [peaks_dataset[f'C{i+1}range'].values for i in range(len(peaks_dataset.chirp))]
                       for j in i])
    bounds = [0] + [peaks_dataset[f'C{i+1}range'].values[-1] for i in range(len(peaks_dataset.chirp))]
    chirp_ind = np.digitize(height, bounds)
    range_ind = argnearest(peaks_dataset[f'C{chirp_ind}range'].values, height)
    time_ind = argnearest(peaks_dataset.time, (time - datetime.datetime(1970, 1, 1)).total_seconds())
    peaks = peaks_dataset[f'C{chirp_ind}{key}'].values[time_ind, range_ind, :]
    peaks = peaks[peaks>0]
    spectrum = spec_dataset[f'C{chirp_ind}Zspec'].values[time_ind, range_ind, :]
    velbins = spec_dataset[f'C{chirp_ind}vel'].values

    fig, ax = plt.subplots(1)
    ax.plot(velbins[peaks], lin2z(spectrum)[peaks], marker='o',
            color='#0339cc', markeredgecolor='k',
            linestyle="None", label='PEAKO peaks', markersize=4)

    ax.plot(velbins, lin2z(spectrum), linestyle='-', linewidth=1, label='raw spectrum')
    ax.set_xlabel('Doppler Velocity [m s$^{-1}$]', fontweight='semibold', fontsize=13)
    ax.set_ylabel('Reflectivity [dBZ m$\\mathregular{^{-1}}$ s]', fontweight='semibold', fontsize=13)
    ax.grid(linestyle=':')
    ax.set_xlim(np.nanmin(velbins), np.nanmax(velbins))
    ax.legend(fontsize=13)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    ax.set_title(f'spectrum at {round(spec_dataset[f"C{chirp_ind}range"].values[range_ind])} m, '
                 f'{format_hms(spec_dataset["time"].values[time_ind])}')
    return fig, ax


def find_index_in_sublist(index, training_index):
    list_of_lengths = [len(i[0]) for i in training_index]
    b = np.cumsum(np.array(list_of_lengths))
    a = 0
    list_of_lengths_2 = list(range(len(list_of_lengths)))
    c = np.digitize(index, b)
    ind = np.where(np.array(list_of_lengths_2) == c)[0]
    left_bound = 0 if c == 0 else b[c-1]
    return int(ind), left_bound


def average_smooth_detect(spec_data, t_avg, h_avg, span, width, prom, all_spectra=False, smoothing_method='loess',
                          max_peaks=5, fill_value=-999.0, **kwargs):
    """
    Average, smooth spectra and detect peaks that fulfill prominence and with criteria.

    :param spec_data:
    :param t_avg: numbers of neighbors in time dimension to average over (on each side).
    :param h_avg: numbers of neighbors in range dimension to average over (on each side).
    :param span: Percentage of number of data points used for smoothing when loess or lowess smoothing is used.
    :param width: minimum peak width in m/s Doppler velocity (width at half-height).
    :param prom: minimum peak prominence in dBZ.
    :param all_spectra: Bool. True if peaks in all spectra should be detected.
    :param smoothing_method:
    :param max_peaks:
    :param fill_value:
    :param kwargs: 'marked_peaks_index', 'procs', 'verbosity'
    :return: peaks: The detected peaks
    """
    avg_spec = average_spectra(spec_data, t_avg, h_avg, **kwargs)
    smoothed_spectra = smooth_spectra(avg_spec, spec_data, span=span, method=smoothing_method)
    peaks = get_peaks(smoothed_spectra, spec_data, prom, width, all_spectra=all_spectra, max_peaks=max_peaks,
                      fill_value=fill_value, **kwargs)
    return peaks


def average_single_bin(specdata_values, B, bin):
    A = specdata_values[:, :, bin]
    C = scipy.signal.convolve2d(A, B, 'same')
    return C


def average_spectra(spec_data, t_avg, h_avg, **kwargs):
    print('averaging...') if 'verbosity' in kwargs and kwargs['verbosity'] > 0 else None
#    processes = kwargs['procs'] if 'procs' in kwargs else 5
    avg_specs_list = []  # initialize empty list
    for f in range(len(spec_data)):
        # average spectra over neighbors in time-height
        avg_specs = spec_data[f].where(spec_data[f].time < 0).load()  # create empty xr data set

        if t_avg == 0 and h_avg == 0:
            avg_specs['doppler_spectrum'][:, :, :] = spec_data[f]['doppler_spectrum'].values[:, :, :]
        else:
            B = np.ones((1+t_avg*2, 1+h_avg*2))/((1+t_avg*2) * (1+h_avg*2))
#                func = partial(average_single_bin, spec_data[f][var_string].data, B)
#                res = pn.map(func, iterable)
#                array_out = np.swapaxes(np.array(res), 0, 2)
#                array_out = array_out.swapaxes(0, 1)
#                avg_specs[var_string][:, :, :] = array_out
            for c in range(len(spec_data[f].chirp)):
                r_ind = [int(i) for i in [0] + list(spec_data[f].rg_offsets.values)][c:c+2]
                for d in range(avg_specs['doppler_spectrum'].values.shape[2]):

                    A = spec_data[f]['doppler_spectrum'].data[:, r_ind[0]:r_ind[1], d]
                    C = scipy.signal.convolve2d(A, B, 'same')
                    avg_specs['doppler_spectrum'][:, r_ind[0]:r_ind[1], d] = C

        avg_specs_list.append(avg_specs)

    return avg_specs_list


def smooth_spectra(averaged_spectra, spec_data, span, method):
    """
    smooth an array of spectra. 'loess' and 'lowess' methods apply a Savitzky-Golay filter to an array.
    Refer to scipy.signal.savgol_filter for documentation on the 1-d filter. 'loess' means that polynomial is
    degree 2; lowess means polynomial is degree 1.
    :param averaged_spectra: list of Datasets of spectra, linear units
    :param spec_data:
    :param span: span used for loess/ lowess smoothing
    :return: spectra_out, an array with same dimensions as spectra containing the smoothed spectra
    """
    print('smoothing...')
    spectra_out = [i.copy(deep=True) for i in averaged_spectra]
    if span == 0.0:
        return spectra_out
    for f in range(len(averaged_spectra)):
        for c in range(len(averaged_spectra[f].chirp)):
            r_ind = [int(i) for i in [0] + list(spec_data[f].rg_offsets.values)][c:c+2]
            velbins = spec_data[f]['velocity_vectors'].values[c, :]
            window_length = round_to_odd(span * len(velbins))
            if method == 'loess':
                spectra_out[f]['doppler_spectrum'].values[:, r_ind[0]: r_ind[1], :] = scipy.signal.savgol_filter(
                    averaged_spectra[f]['doppler_spectrum'].values[:, r_ind[0]: r_ind[1], :], window_length,
                    polyorder=2, axis=2, mode='nearest')
            elif method == 'lowess':
                spectra_out[f]['doppler_spectrum'].values[:, r_ind[0]: r_ind[1], :] = scipy.signal.savgol_filter(
                    averaged_spectra[f]['doppler_spectrum'].values[:, r_ind[0]: r_ind[1], :],
                    window_length,
                    polyorder=1, axis=2,
                    mode='nearest')
    return spectra_out


def get_peaks(spectra, spec_data, prom, width_thresh, all_spectra=False, max_peaks=5, fill_value=-999, **kwargs):
    """
    detect peaks in (smoothed) spectra which fulfill minimum prominence and width criteria.
    :param spec_data
    :param spectra: list of data arrays containing (smoothed) spectra in linear units
    :param prom: minimum prominence in dbZ
    :param width_thresh: width threshold in m/s
    :param all_spectra: Bool. True if peaks in all the spectra should be detected.
    :param kwargs: marked_peaks_index
    :return: peaks: list of data arrays containing detected peak indices. Length of this list is the same as the
    length of the spectra (input parameter) list.
    """
    print('detecting...') if 'verbosity' in kwargs and kwargs['verbosity'] > 0 else None
    peaks = []
    procs = kwargs['procs'] if 'procs' in kwargs else 5
    for f in range(len(spectra)):
        peaks_dataset = xr.Dataset()
        # create an xarray dataset and add one empty data array to it
        peaks_array = xr.Dataset(data_vars={'PeakoPeaks': xr.DataArray(np.full(
            (spectra[f]['doppler_spectrum'].values.shape[0:2] +
             (max_peaks,)), np.nan, dtype=np.int),
            dims=['time', 'range', 'peaks'],
            coords=[spectra[f]['time'], spectra[f]['range'],
                    xr.DataArray(range(max_peaks))])})
        for c in range(len(spectra[f].chirp)):
            # convert width_thresh units from m/s to # of Doppler bins:
            width_thresh = width_thresh/np.nanmedian(np.diff(spec_data[f]['velocity_vectors'].values[c, :]))

            r_ind = [int(i) for i in [0] + list(spec_data[f].rg_offsets.values)][c:c + 2]
            if all_spectra:

                peaks_all_spectra = xr.apply_ufunc(peak_detection_dask, spectra[f]['doppler_spectrum'][:,
                                                                        r_ind[0]: r_ind[1], :],
                                                   prom, fill_value, width_thresh, max_peaks, dask='parallelized')
                peaks_array['PeakoPeaks'].data[:, r_ind[0]: r_ind[1], :] = peaks_all_spectra.data[:, :, 0:max_peaks]

            else:
                assert 'marked_peaks_index' in kwargs, "if param all_spectra is set to False, you have to supply " \
                                                       "marked_peaks_index as key word argument"
                marked_peaks_index = kwargs['marked_peaks_index']
                t_ind, h_ind = np.where(marked_peaks_index[f][:, r_ind[0]: r_ind[1]] == 1)

                # loop over height-time combinations where spectra were marked
                iterable = [[t, h] for t, h in zip(t_ind, h_ind)]
                pn = Pool(processes=procs)
                func = partial(peak_detection_multiprocessing, spectra[f]['doppler_spectrum'].data[:, r_ind[0]: r_ind[1],
                                                                 :], prom, fill_value, width_thresh, max_peaks)
                res = pn.map(func, iterable)
                for i, j in enumerate(res):
                    t, h = iterable[i]
                    peaks_array['PeakoPeaks'].data[t, r_ind[0] + h, 0:len(j)] = j

        # update the dataset (add the peaks_array dataset)
        peaks_dataset.update(other=peaks_array)
        # add chirps
        peaks_dataset = peaks_dataset.assign({'chirp': spectra[f].chirp})
        # and append it to the list
        peaks.append(peaks_dataset)

    return peaks


def peak_detection_dask(numpy_array, prom, fill_value, width_thresh, max_peaks):
    spectra = lin2z(numpy_array)
    fillvalue = np.ma.filled(np.nanmin(spectra, axis=2)[:, :, np.newaxis], -100.)
    spectra = np.ma.filled(spectra, fillvalue)
    out = np.apply_along_axis(detect_single_spectrum, 2, spectra, prom=prom, fill_value=fill_value,
                        width_thresh=width_thresh, max_peaks=max_peaks)
    return out


def peak_detection_multiprocessing(spectra, prom, fill_value, width_thresh, max_peaks, th_ind):
    t, h = th_ind
    spectrum = spectra[t, h, :]
    spectrum = lin2z(spectrum)
    spectrum.data[spectrum.mask] = np.nanmin(spectrum)
    spectrum = spectrum.data
    locs = detect_single_spectrum(spectrum, prom, fill_value, width_thresh, max_peaks)
    locs = locs[0:max_peaks]
    return locs


def detect_single_spectrum(spectrum, prom, fill_value, width_thresh, max_peaks):

    # call scipy.signal.find_peaks to detect peaks in the (logarithmic) spectrum
    # it is important that nan values are not included in the spectrum passed to si
    locs, props = si.find_peaks(spectrum, prominence=prom)
    # find left and right edges of peaks
    le, re = find_edges(spectrum, fill_value, locs)
    # compute the width
    width = peak_width(spectrum, locs, le, re)
    locs = locs[width > width_thresh]
    locs = locs[0: max_peaks] if len(locs) > max_peaks else locs
    # TODO
    #  Murks : artificially create output dimension of same length as Doppler bins to avoid xarray value error
    out = np.full(spectrum.shape[0], np.nan, dtype=int)
    out[range(len(locs))] = locs
    return out


class Peako(object):
    def __init__(self, training_data=[], peak_detection='peako', optimization_method='loop',
                 smoothing_method='loess', max_peaks=5, k=0, verbosity=0, **kwargs):

        """
        initialize a Peako object
        :param training_data: list of strings (netcdf files to read in written by mark_peaks module,
        filenames starting with marked_peaks_...)
        :param peak_detection: method for peak detection. Only option right now is 'peako'; later 'peakTree' option
        is to be added. Default is 'peako'.
        :param optimization_method: Either 'loop' or 'scipy'. In case of 'loop' looping over different parameter
        combinations is performed in a brute-like way. Option 'scipy' uses differential evolution toolkit to find
        optimal solution. Default is 'loop'.
        :param smoothing_method: method for smoothing Doppler spectra. Options are 'loess', 'lowess' (both included
        in scipy.signal). The plan is to also add peakTree smoothing using convolution with a defined window function.
        The default is 'loess' smoothing.
        :param max_peaks: maximum number of peaks to be detected by the algorithm. Defaults to 5.
        :param k: integer specifying parameter k in k-fold cross-validation. If it's set to 0 (the default), the
        training data is not split. If it's different from 0, training data is split into k subsets (folds), where
        each fold will be used as the test set one time.
        :param verbosity: level of how much detail is printed into the console (debugging info)
        :param kwargs 'training_params' = dictionary containing 't_avg', 'h_avg', 'span', 'width' and 'prom' values over
        which to loop if optimization_method is set to 'loop'. 'procs': number of processes to use (multiprocessing
        library)
        """
        self.training_files = training_data
        self.training_data = [xr.open_mfdataset(fin, combine='by_coords') for fin in training_data]
        self.specfiles = ['/'.join(f.split('/')[:-1]) + '/' + f.split('/')[-1][13:] for f in self.training_files]
        self.spec_data = [xr.open_mfdataset(fin, combine='by_coords') for fin in self.specfiles]
        self.peak_detection_method = peak_detection
        self.optimization_method = optimization_method
        self.smoothing_method = smoothing_method
        self.marked_peaks_index = []
        self.max_peaks = max_peaks
        self.k = k+1
        self.current_k = 0
        self.k_fold_cv = False if k == 0 else True
        self.fill_value = self.spec_data[0]._FillValue
        self.training_params = kwargs['training_params'] if 'training_params' in kwargs else \
            {'t_avg': range(2), 'h_avg': range(2), 'span': np.arange(0.005, 0.02, 0.005),
             'width': np.arange(0, 1.5, 0.5), 'prom': np.arange(0, 1.5, 0.5)}
        self.training_result = {'loop': [np.empty((1, 6))], 'scipy': [np.empty((1, 6))]} if not self.k_fold_cv else {
            'loop': [np.empty((1, 6))]*self.k, 'scipy': [np.empty((1, 6))]*self.k}
        self.peako_peaks_training = {'loop': [[] for _ in range(self.k)], 'scipy': [[] for _ in range(self.k)]}
        self.peako_peaks_testing = {'loop': [], 'scipy': []}
        self.testing_files = []
        self.testing_data = []
        self.specfiles_test = []
        self.spec_data_test = []
        self.plot_dir = kwargs['plot_dir'] if 'plot_dir' in kwargs else ''
        self.verbosity = verbosity
        self.procs = kwargs['procs'] if 'procs' in kwargs else 5

    def create_training_mask(self):
        """
        Find the entries in Peako.training_data that have values stored in them, i.e. the indices of spectra with
        user-marked peaks. Store this mask in Peako.marked_peaks_index.
        """
        self.marked_peaks_index = []
        list_out = []
        for f in range(len(self.training_data)):
            list_out.append(xr.DataArray(~(self.training_data[f]['peaks'].values[:, :, 0] == self.fill_value)*1,
                                          dims=['range', 'time']))
        if not self.k_fold_cv:
            self.marked_peaks_index = list_out
        else:
            return list_out

    def train_peako(self):
        if not self.k_fold_cv:
            # locate the spectra that were marked by hand
            self.create_training_mask()
            result = self.train_peako_inner()
            return result
        else:
            result_list_out = []
            from sklearn.model_selection import KFold
            # split the training data into k subsets and use one as a testing set
            training_mask = self.create_training_mask()
            empty_training_mask = copy.deepcopy(training_mask)
            for i1 in range(len(empty_training_mask)):
                empty_training_mask[i1].values[:, :] = 0

            training_index = [np.where(training_mask[f] == 1)
                      for f in range(len(training_mask))]


            list_of_lengths = [len(i[0]) for i in training_index]
            num_marked_spectra = np.sum(list_of_lengths)
            cv = KFold(n_splits=self.k, random_state=42, shuffle=True)
            self.current_k = 0
            for index_training, index_testing in cv.split(range(num_marked_spectra)):
                this_training_mask = copy.deepcopy(empty_training_mask)
                if self.verbosity > 0:
                    print("k: ", self.current_k, "\n")
                    print("Train Index: ", index_training, "\n")
                    print("Test Index: ", index_testing)
                for f in range(len(index_training)):
                    i_1, left_bound = find_index_in_sublist(index_training[f], training_index)
                    this_training_mask[i_1].values[training_index[i_1][0][index_training[f]-left_bound],
                                                        training_index[i_1][1][index_training[f]-left_bound]] = 1
                    # possibly missing a +/- 1 here
                self.marked_peaks_index = this_training_mask
                result = self.train_peako_inner()
                result_list_out.append(result)
                self.current_k += 1

            self.marked_peaks_index = training_mask
            return result_list_out

    def train_peako_inner(self):
        """
        Train the peak finding algorithm.
        Depending on Peako.optimization_method, looping over possible parameter combinations or an optimization toolkit
        is used to find the combination of time and height averaging, smoothing span, minimum peak width and minimum
        peak prominence which yields the largest similarity between user-found and algorithm-detected peaks.

        """

        # if optimization method is set to "loop", loop over possible parameter combinations:
        if self.optimization_method == 'loop':
            similarity_array = np.full([len(self.training_params[key]) for key in self.training_params.keys()], np.nan)
            for i, t_avg in enumerate(self.training_params['t_avg']):
                for j, h_avg in enumerate(self.training_params['h_avg']):
                    for k, span in enumerate(self.training_params['span']):
                        for l, wth in enumerate(self.training_params['width']):
                            for m, prom in enumerate(self.training_params['prom']):
                                if self.verbosity > 0:
                                    print(f'finding peaks for t={t_avg}, h={h_avg}, span={span}, width={wth}, '
                                          f'prom={prom}')
                                peako_peaks = average_smooth_detect(spec_data=self.spec_data, t_avg=t_avg, h_avg=h_avg,
                                                                    span=span, width=wth, prom=prom,
                                                                    smoothing_method=self.smoothing_method,
                                                                    max_peaks=self.max_peaks,
                                                                    fill_value=self.fill_value,
                                                                    marked_peaks_index=self.marked_peaks_index,
                                                                    procs=self.procs, verbosity=self.verbosity)
                                # compute the similarity
                                similarity = self.area_peaks_similarity(peako_peaks, array_out=False)
                                similarity_array[i, j, k, l, m] = similarity
                                self.training_result['loop'][self.current_k] = np.append(self.training_result['loop'][
                                    self.current_k], [[t_avg, h_avg, span, wth, prom, similarity]],
                                                                         axis=0)
                                if self.verbosity > 0:
                                    print(f"similarity: {similarity}, t:{t_avg}, h:{h_avg}, span:{span}, width:{wth}, "
                                          f"prom:{prom}")

            # remove the first line from the training result
            self.training_result['loop'][self.current_k] = np.delete(self.training_result['loop'][self.current_k], 0,
                                                                     axis=0)

            # extract the parameter combination yielding the maximum in similarity
            t, h, s, w, p = np.unravel_index(np.argmax(similarity_array, axis=None), similarity_array.shape)
            return {'t_avg': self.training_params['t_avg'][t],
                    'h_avg': self.training_params['h_avg'][h],
                    'span': self.training_params['span'][s],
                    'width': self.training_params['width'][w],
                    'prom': self.training_params['prom'][p],
                    'similarity': np.max(similarity_array)}

        elif self.optimization_method == 'scipy':
            bounds = [(min(self.training_params['t_avg']), max(self.training_params['t_avg'])),
                      (min(self.training_params['h_avg']), max(self.training_params['h_avg'])),
                      (np.log10(min(self.training_params['span'])), np.log10(max(self.training_params['span']))),
                      (min(self.training_params['width']), max(self.training_params['width'])),
                      (min(self.training_params['prom']), max(self.training_params['prom']))]
            result = differential_evolution(self.fun_to_minimize, bounds=bounds)

            # remove the first line from the training result
            self.training_result['scipy'][self.current_k] = np.delete(self.training_result['scipy'][self.current_k], 0,
                                                                      axis=0)
            return result

    def fun_to_minimize(self, parameters):
        """
        Function which is minimized by the optimization toolkit (differential evolution).
        It averages the neighbor spectra in a range defined by t_avg and h_avg,
        calls smooth_spectrum with the defined method (Peako.smoothing_method),
        and calls get_peaks using the defined prominence and width. The t_avg, h_avg, span, width and prominence
        parameters are passed as parameters:

        :param parameters: list containing t_avg, h_avg, span, width and prominence. If this function is called within
        scipy.optimize.differential_evolution, this corresponds to the order of the elements in "bounds"
        :return: res: Result (negative similarity measure based on area below peaks); negative because optimization
        toolkits usually search for the minimum.

        """

        t_avg, h_avg, span, width, prom = parameters
        # trick to search for span in a larger (logarithmic) search space
        span = 10 ** span
        # trick to get integers:
        t_avg = np.int(round(t_avg))
        h_avg = np.int(round(h_avg))

        peako_peaks = average_smooth_detect(self.spec_data, t_avg=t_avg, h_avg=h_avg, span=span, width=width, prom=prom,
                                            smoothing_method=self.smoothing_method, max_peaks=self.max_peaks,
                                            fill_value=self.fill_value, marked_peaks_index=self.marked_peaks_index,
                                            procs=self.procs, verbosity=self.verbosity)
        # compute the similarity
        res = self.area_peaks_similarity(peako_peaks, array_out=False)

        # print(",".join(map(str, np.append(parameters, res))))
        self.training_result['scipy'][self.current_k] = np.append(self.training_result['scipy'][self.current_k],
                                                                  [[t_avg, h_avg, span, width, prom, res]], axis=0)

        # for differential evolution, return negative similarity:
        return -res


    def area_peaks_similarity(self, algorithm_peaks, array_out=False):
        """ Compute similarity measure based on overlapping area of hand-marked peaks by a user and algorithm-detected
            peaks in a radar Doppler spectrum

            :param algorithm_peaks: ndarray of indices of spectrum where peako detected peaks
            :param array_out: Bool. If True, area_peaks_similarity will return a list of xr.Datasets containing the
            computed similarities for each spectrum in the time-height grid. If False, the integrated similarity (sum)
            of all the hand-marked spectra is returned. Default is False.

        """
        sim_out = [] if array_out else 0
        print('computing similarity...') if self.verbosity > 0 else None
        # loop over files and chirps, and then over the spectra which were marked by hand
        for f in range(len(self.specfiles)):
            for c in range(len(self.training_data[f].chirp)):
                velbins = self.spec_data[f]['velocity_vectors'].values[c, :]
                r_ind = [int(i) for i in [0] + list(self.spec_data[f].rg_offsets.values)][c:c + 2]
                t_ind, h_ind = np.where(self.marked_peaks_index[f][:, r_ind[0]: r_ind[1]] == 1)
                for h, t in zip(h_ind, t_ind):
                    user_peaks = self.training_data[f]['peaks'].values[t, r_ind[0] + h, :]
                    user_peaks = user_peaks[~(user_peaks == self.fill_value)]
                    # convert velocities to indices
                    user_peaks = np.asarray([argnearest(velbins, val) for val in user_peaks])
                    spectrum = self.spec_data[f]['doppler_spectrum'].values[t, r_ind[0] + h, :]
                    #spectrum[spectrum == self.fill_value] = np.nan
                    spectrum_db = lin2z(spectrum).filled(0.0)
                    spectrum_db[spectrum == self.fill_value] = 0.0
                    user_peaks.sort()
                    peako_peaks = algorithm_peaks[f]['PeakoPeaks'].values[t, r_ind[0] + h, :]
                    peako_peaks = peako_peaks[peako_peaks > 0]
                    peako_peaks.sort()
                    le_user_peaks, re_user_peaks = find_edges(spectrum, self.fill_value, user_peaks)
                    le_alg_peaks, re_alg_peaks = find_edges(spectrum, self.fill_value, peako_peaks)
                    similarity = 0
                    overlap_area = math.inf
                    while(len(peako_peaks) > 0) & (len(user_peaks) > 0) & (overlap_area > 0):
                        # compute maximum overlapping area
                        user_ind, alg_ind, overlap_area = overlapping_area([le_user_peaks, re_user_peaks],
                                                                           [le_alg_peaks, re_alg_peaks],
                                                                           spectrum_db, np.nanmin(spectrum_db), velbins)
                        similarity = similarity + overlap_area
                        if user_ind is not None:
                            user_peaks = np.delete(user_peaks, user_ind)
                            le_user_peaks = np.delete(le_user_peaks, user_ind)
                            re_user_peaks = np.delete(re_user_peaks, user_ind)
                        if alg_ind is not None:
                            peako_peaks = np.delete(peako_peaks, alg_ind)
                            le_alg_peaks = np.delete(le_alg_peaks, alg_ind)
                            re_alg_peaks = np.delete(re_alg_peaks, alg_ind)

                    # Subtract area of non-overlapping regions
                    for i in range(len(le_alg_peaks)):
                        similarity = similarity - area_above_floor(le_alg_peaks[i], re_alg_peaks[i], spectrum_db,
                                                                   np.nanmin(spectrum_db), velbins)
                    for i in range(len(le_user_peaks)):
                        similarity = similarity - area_above_floor(le_user_peaks[i], re_user_peaks[i], spectrum_db,
                                                                   np.nanmin(spectrum_db), velbins)
                    #print(user_peaks, algorithm_peaks, similarity)

                    if not array_out:
                        sim_out += similarity
        return sim_out

    def assert_training(self):
        """
        assertion that training has happened. Checks if there is a training mask in Peako.marked_peaks_index and that
        there is a training result stored in Peako.training_result.
        """

        # assert that training has happened
        # check if there is a training mask and if there is a result
        assert(len(self.marked_peaks_index[0]) > 0), "no training mask available"
        assert(self.training_result['loop'][0].shape[0] + self.training_result['scipy'][0].shape[0] > 2), \
            "no training result"

    def check_store_found_peaks(self):
        """
        check if peak locations for optimal parameter combination have been stored, if not store them.
        """

        # for each of the optimization methods, check if there is a result in Peako.training_result
        for j in self.training_result.keys():
            for k in range(len(self.training_result[j])):
                if self.training_result[j][k].shape[0] > 1:
                # if there is a result, extract the optimal parameter combination height averages, time averages, span,
                # width, and prominence threshold
                    i_max = np.argmax(self.training_result[j][k][:, -1])
                    t, h, s, w, p = self.training_result[j][k][i_max, :-1]
                # if there are no peaks stored in Peako.peako_peaks_training, find the peaks for each spectrum in
                # the training files
                    if len(self.peako_peaks_training[j][k]) == 0:
                        print('finding peaks for all times and ranges...')
                        self.peako_peaks_training[j][k] = average_smooth_detect(self.spec_data, t_avg=int(t),
                                                                                h_avg=int(h), span=s, width=w, prom=p,
                                                                                all_spectra=True,
                                                                                smoothing_method=self.smoothing_method,
                                                                                max_peaks=self.max_peaks,
                                                                                fill_value=self.fill_value,
                                                                                procs=self.procs)

                    # or if the shape of the training data does not match the shape of the stored found peaks
                    elif self.peako_peaks_training[j][k][0]['PeakoPeaks'].values.shape[:2] != \
                            self.spec_data[0]['doppler_spectrum'].shape[:2]:
                        print('finding peaks for all times and ranges...')

                        self.peako_peaks_training[j][k] = average_smooth_detect(self.spec_data, t_avg=int(t),
                                                                                h_avg=int(h), span=s, width=w, prom=p,
                                                                                smoothing_method=self.smoothing_method,
                                                                                max_peaks=self.max_peaks,
                                                                                fill_value=self.fill_value,
                                                                                all_spectra=True, procs=self.procs)

    def training_stats(self, make_3d_plots=False, **kwargs):
        """
        print out training statistics
        :param make_3d_plots: bool: Default is False. If set to True, plot_3d_plots will be called
        :param kwargs: k: number of subset (if k-fold cross-validation is used) for which statistics should be returned.
         Defaults to 0
         """

        self.assert_training()
        k = kwargs['k'] if 'k' in kwargs else 0
        # compute maximum possible similarity
        user_peaks = []
        for f in range(len(self.specfiles)):
            peaks_dataset = xr.Dataset()
            peaks_array = xr.Dataset(data_vars={'PeakoPeaks': xr.DataArray(np.full(
                self.training_data[f]['peaks'].values.shape,
                np.nan, dtype=np.int), dims=['time', 'range', 'peaks'])})
            for c in range(len(self.training_data[f].chirp)):
                velbins = self.spec_data[f]['velocity_vectors'].values[c, :]

                r_ind = [int(i) for i in [0] + list(self.spec_data[f].rg_offsets.values)][c:c + 2]
                # convert m/s to indices (call vel_to_ind)
                t_ind, h_ind = np.where(self.marked_peaks_index[f][:, r_ind[0]: r_ind[1]] == 1)
                for h, t in zip(h_ind, t_ind):
                    indices = vel_to_ind(self.training_data[f]['peaks'].values[t, r_ind[0] + h, :], velbins,
                                         self.fill_value)
                    peaks_array['PeakoPeaks'].values[t, r_ind[0] + h, :] = indices
            peaks_dataset.update(other=peaks_array)
            user_peaks.append(peaks_dataset)

        maximum_similarity = self.area_peaks_similarity(user_peaks)
        for j in self.training_result.keys():
            if self.training_result[j][k].shape[0] > 1:
                print(f'{j}, k={k}:')
                catch = np.nanmax(self.training_result[j][k][:, -1])
                print(f'similarity is {round(catch/maximum_similarity*100,2)}% of maximum possible similarity')
                print('h_avg: {0[0]}, t_avg:{0[1]}, span:{0[2]}, width: {0[3]}, prom: {0[4]}'.format(
                    (self.training_result[j][k][np.argmax(self.training_result[j][k][:, -1]), :-1])))

                if make_3d_plots:
                    fig, ax = self.plot_3d_plots(j, k=k)
                    if 'k' in kwargs:
                        fig.suptitle(f'{j}, k = {k}')
                    if len(self.plot_dir) > 0:
                        fig.savefig(self.plot_dir + f'3d_plot_{j}_k{k}.png')
        return maximum_similarity

    def plot_3d_plots(self, key, k=0):
        """
        Generates 4 panels of 3D plots of parameter vs. parameter vs. similarity for evaluating the training of pyPEAKO
        by eye

        :param key: dictionary key in Peako.training_result for which to make the 3D plots, either 'loop' or 'scipy'.
        :return: fig, ax : matplotlib.pyplot figure and axes
        """

        from mpl_toolkits.mplot3d import Axes3D

        training_result = self.training_result[key][k]
        fig, ax = plt.subplots(2, 2, subplot_kw=dict(projection='3d'))
        ax[0, 0].scatter(training_result[:, 0], training_result[:, 1], training_result[:, -1], zdir='z',
                         c=training_result[:, -1], cmap='seismic')
        ax[0, 0].set_xlabel('height averages')
        ax[0, 0].set_ylabel('time averages')
        ax[0, 0].set_zlabel('similarity')

        ax[1, 1].scatter(training_result[:, 3], training_result[:, 2], training_result[:, -1], zdir='z',
                         c=training_result[:, -1], cmap='seismic')
        ax[1, 1].set_xlabel('width')
        ax[1, 1].set_ylabel('span')
        ax[1, 1].set_zlabel('similarity')

        ax[0, 1].scatter(training_result[:, 4], training_result[:, 3], training_result[:, -1], zdir='z',
                         c=training_result[:, -1], cmap='seismic')
        ax[0, 1].set_xlabel('prom')
        ax[0, 1].set_ylabel('width')
        ax[0, 1].set_zlabel('similarity')

        ax[1, 0].scatter(training_result[:, 4], training_result[:, 1], training_result[:, -1], zdir='z',
                         c=training_result[:, -1], cmap='seismic')
        ax[1, 0].set_xlabel('prom')
        ax[1, 0].set_ylabel('time averages')
        ax[1, 0].set_zlabel('similarity')

        return fig, ax

    def plot_user_algorithm_spectrum(self, **kwargs):
        """
        Plot a cloud radar Doppler spectrum along with the user-marked peaks and the algorithm-detected peaks for each
        of the training results in the Peako.peako_peaks_training dictionary.

        :param kwargs: 'seed' : set seed to an integer number for reproducibility
        :return:
        """

        # assert that training has happened:
        self.assert_training()

        # set random seed if it's given in the key word arguments:
        if 'seed' in kwargs:
            random.seed(kwargs['seed'])
        k = kwargs['k'] if 'k' in kwargs else 0

        # select a random user-marked spectrum
        f = random.randint(0, len(self.marked_peaks_index) - 1)
        # this can be problematic if there are no marked spectra for one of the chirps...
        t_ind, h_ind = np.where(self.marked_peaks_index[f] == 1)

        i = random.randint(0, len(h_ind) - 1)
        c = np.digitize(h_ind[i], [0] + list(self.spec_data[f].rg_offsets.values))
        velbins = self.spec_data[f]['velocity_vectors'].values[c, :]
        spectrum = self.spec_data[f]['doppler_spectrum'].values[t_ind[i], h_ind[i], :]
        user_ind = vel_to_ind(self.training_data[f]['peaks'].values[t_ind[i], h_ind[i], :], velbins=velbins,
                              fill_value=self.fill_value)
        user_ind = user_ind[user_ind > 0]

        # call check_store_found_peaks to make sure that there is peaks in Peako.peako_peaks_training
        self.check_store_found_peaks()

        # plotting
        fsz = 13
        star = matplotlib.path.Path.unit_regular_star(6)
        circle = matplotlib.path.Path.unit_circle()
        # concatenate the circle with an internal cutout of the star
        verts = np.concatenate([circle.vertices, star.vertices[::-1, ...]])
        codes = np.concatenate([circle.codes, star.codes])
        cut_star = matplotlib.path.Path(verts, codes)

        fig, ax = plt.subplots(1)
        ax.plot(velbins, lin2z(spectrum), linestyle='-', linewidth=1, label='raw spectrum')

        c_ind = 0
        for j in self.peako_peaks_training.keys():
            print(f'{j}, k:{k}')
            if self.training_result[j][k].shape[0] > 1:
                peako_ind = self.peako_peaks_training[j][k][f]['PeakoPeaks'].values[t_ind[i], h_ind[i], :]
                peako_ind = peako_ind[peako_ind > 0]

                ax.plot(velbins[peako_ind], lin2z(spectrum)[peako_ind], marker='o',
                        color=['#0339cc', '#0099ff', '#9933ff'][c_ind], markeredgecolor='k',
                        linestyle="None", label=f'PEAKO peaks {j}', markersize=[8, 7, 6][c_ind])
                c_ind += 1

        ax.plot(velbins[user_ind], lin2z(spectrum)[user_ind], marker=cut_star, color='r',
                linestyle="None", label='user peaks')
        ax.set_xlabel('Doppler Velocity [m s$^{-1}$]', fontweight='semibold', fontsize=fsz)
        ax.set_ylabel('Reflectivity [dBZ m$\\mathregular{^{-1}}$ s]', fontweight='semibold', fontsize=fsz)
        ax.grid(linestyle=':')
        ax.set_xlim(np.nanmin(velbins), np.nanmax(velbins))
        ax.legend(fontsize=fsz)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        ax.set_title(f'spectrum at {round(self.spec_data[f]["range"].values[h_ind[i]])} m, '
                     f'{format_hms(self.spec_data[f]["time"].values[int(t_ind[i])])}')
        if len(self.plot_dir) > 0:
            fig.savefig(self.plot_dir + f'spectrum_{round(self.spec_data[f]["range"].values[h_ind[i]])}m'
                                        f'_{format_hms(self.spec_data[f]["time"].values[int(t_ind[i])])}_k{k}.png')

        return fig, ax

    def plot_algorithm_spectrum(self,  file, time, height, mode='training'):
        """
        :param file: number of file (integer)
        :param time: the time of the spectrum to plot (datetime.datetime)
        :param height: the range of the spectrum to plot (km)
        :param mode: 'training', 'testing'
        :return:
        """
        pass

    def test_peako(self, test_data, **kwargs):
        """
        Add testing data to the Peako object and print out some stats for the testing data set

        :param test_data: list of netcdf files with hand-marked peaks
        :param kwargs: 'seed' to pass on to Peako.plot_user_algorithm_spectrum
        """
        self.testing_files = test_data
        self.testing_data = [xr.open_mfdataset(fin, combine='by_coords') for fin in test_data]
        self.specfiles_test = ['/'.join(f.split('/')[:-1]) + '/' + f.split('/')[-1][13:] for f in self.testing_files]
        self.spec_data_test = [xr.open_mfdataset(fin, combine='by_coords') for fin in self.specfiles_test]

        # TODO !!Murks!! Find better solution for this
        # copy the Peako object and replace training data with testing data to utilize the functions using spec_data
        q = copy.deepcopy(self)
        q.training_files = test_data
        q.training_data = self.testing_data
        q.specfiles = self.specfiles_test
        q.spec_data = self.spec_data_test
        q.marked_peaks_index = []
        q.create_training_mask()
        q.training_stats()
        q.plot_user_algorithm_spectrum(**kwargs)
        self.peako_peaks_testing = q.peako_peaks_training

    def plot_numpeaks_timeheight(self, mode='training', **kwargs):
        """
        Plot time-height plots of the number of found peaks by peako (for different optimization results if they are
        available) and of the peaks marked by a human user, for each of the files in the list Peako.training_data or
        Peako.testing_data

        :param mode: (string) Either 'training' or 'testing'
        :return:
        """

        if mode == 'training':
            # call check_store_found_peaks to make sure that there is peaks in Peako.peako_peaks_training
            self.check_store_found_peaks()
            algorithm_peaks = self.peako_peaks_training
            user_peaks = self.training_data
        elif mode == 'testing':
            algorithm_peaks = self.peako_peaks_testing
            user_peaks = self.testing_data
        elif mode == 'manual':
            assert 'peako_params' in kwargs, 'peako_params (list of five parameters) must be supplied'
            t, h, s, w, p = kwargs['peako_params']
            algorithm_peaks = {'manual': average_smooth_detect(self.spec_data, t_avg=int(t), h_avg=int(h), span=s,
                                                               width=w, prom=p, smoothing_method=self.smoothing_method,
                                                               fill_value=self.fill_value, max_peaks=self.max_peaks,
                                                               all_spectra=True, procs=self.procs)}
            self.create_training_mask()
            user_peaks = self.training_data
        # plot number of peako peaks for each of the training files and each of the optimization methods,
        # and number of user-found peaks
        for j in algorithm_peaks.keys():
            # loop over the files
            for f in range(len(algorithm_peaks[j])):
                fig, ax = plot_timeheight_numpeaks(algorithm_peaks[j][f], key='PeakoPeaks')
                ax.set_title(f'{mode}, optimization: {j}, file number {f+1}')
                if len(self.plot_dir) > 0:
                    fig.savefig(self.plot_dir + f'{mode}_{f+1}_height_time_peako_{j}.png')
        for f in range(len(user_peaks)):
            fig, ax = plot_timeheight_numpeaks(user_peaks[f], key='peaks')
            ax.set_title(f'{mode}, user peaks, file number {f+1}')
            if len(self.plot_dir) > 0:
                fig.savefig(self.plot_dir + f'{mode}_{f+1}_height_time_user.png')

    def plot_cfad(self):
        """
        plot contoured frequency by altitude diagram (CFAD)

        """
        pass


class TrainingData(object):
    def __init__(self, specfiles_in, num_spec=[30], max_peaks=5):
        """
        :param specfiles_in: list of radar spectra files (netcdf format)
        :param num_spec: (list) number of spectra to mark by the user (default 30)
        :param max_peaks: (int) maximum number of peaks per spectrum (default 5)

        """
        self.specfiles_in = specfiles_in
        self.spec_data = [xr.open_mfdataset(fin, combine='by_coords') for fin in specfiles_in]
        self.num_spec = []
        self.tdim = []
        self.rdim = []
        self.training_data_out = []
        self.peaks_ncfiles = []
        self.plot_count = []
        self.fill_value = self.spec_data[0]._FillValue

        for i in range(len(self.spec_data)):
            self.num_spec.append(num_spec[0])
            num_spec.append(num_spec.pop(0))
        self.max_peaks = max_peaks
        self.update_dimensions()

    def add_spectrafile(self, specfile, num_spec=30):
        """
         Open another netcdf file and add it to the list of TrainingData.spec_data
        :param specfile: (str)  spectra netcdf file to add the list of training data
        :param num_spec: (int)  number of spectra to mark by the user (default is 30)
        """
        self.spec_data.append(xr.open_mfdataset(specfile, combine='by_coords'))
        self.num_spec.append(num_spec)
        self.update_dimensions()

    def update_dimensions(self):
        """
        update the list of time and range dimensions stored in TrainingData.tdim and TrainingData.rdim
        update arrays in which found peaks are stored
        also update the names of the netcdf files into which found peaks are stored
        """
        # overwrite dimensions with empty lists
        self.tdim = []
        self.rdim = []
        self.training_data_out = []

        # loop over netcdf files
        for f in range(len(self.spec_data)):
            self.tdim.append(len(self.spec_data[f]['time']))
            self.rdim.append(len(self.spec_data[f]['range']))
            self.training_data_out.append(np.full((self.tdim[-1], self.rdim[-1], self.max_peaks), self.fill_value))
            ncfile = '/'.join(self.specfiles_in[f].split('/')[0:-1]) + \
                     '/' + 'marked_peaks_' + self.specfiles_in[f].split('/')[-1]
            self.peaks_ncfiles.append(ncfile)
            self.plot_count.append(0)

    def mark_random_spectra(self, **kwargs):
        """
        Mark random spectra in TrainingData.spec_data (number of randomly drawn spectra in time-height space defined by
        TrainingData.num_spec) and save x and y locations
        :param kwargs:
               num_spec: update TrainingData.num_spec

        """

        if 'num_spec' in kwargs:
            self.num_spec[:] = kwargs['num_spec']
        for n in range(len(self.spec_data)):
            s = 0
            while s < self.num_spec[n]:
                random_index_t = random.randint(1, self.tdim[n]-1)
                random_index_r = random.randint(1, self.rdim[n]-1)
                print(f'r: {random_index_r}, t: {random_index_t}')
                vals, powers = self.input_peak_locations(n, random_index_t, random_index_r)
                if not np.all(np.isnan(vals)):
                    self.training_data_out[n][random_index_t, random_index_r, 0:len(vals)] = vals
                    s += 1
                    self.plot_count[n] = s

    def input_peak_locations(self, n_file, t_index, r_index):
        """
        :param n_file: the index of the netcdf file from which to mark spectrum by hand
        :param t_index: the time index of the spectrum
        :param r_index: the range index of the spectrum
        :return peakVals: The x values (in units of Doppler velocity) of the marked peaks
        :return peakPowers: The y values (in units of dBZ) of the marked peaks
        """

        #matplotlib.use('TkAgg')
        peakVals = []
        peakPowers = []
        n_rg = self.spec_data[n_file]['rg_offsets']
        c_ind = np.digitize(r_index, n_rg)
        #print(f'range index {r_index} is in chirp {c_ind} with ranges in chirps {n_rg[1:]}')

        heightindex_center = r_index # - n_rg[c_ind - 1]
        timeindex_center = t_index
        this_spectrum_center = self.spec_data[n_file]['doppler_spectrum'][int(timeindex_center), int(heightindex_center),
                                                                        :]

        #print(f'time index center: {timeindex_center}, height index center: {heightindex_center}')
        if not np.sum(~(this_spectrum_center.values == self.spec_data[n_file].attrs['_FillValue'])) < 2 and \
                not np.sum(~np.isnan(this_spectrum_center.values)) < 2:
            # if this spectrum is not empty, we plot 3x3 panels with shared x and y axes
            fig, ax = plt.subplots(3, 3, figsize=[11, 11], sharex=True, sharey=True)
            fig.suptitle(f'Mark peaks in spectrum in center panel. Fig. {self.plot_count[n_file]+1} out of '
                         f'{self.num_spec[n_file]}; File {n_file+1} of {len(self.spec_data)}')
            for dim1 in range(3):
                for dim2 in range(3):
                    if not (dim1 == 1 and dim2 == 1):  # if this is not the center panel plot
                        comment = ''
                        heightindex = r_index - 1 + dim1
                        timeindex = t_index - 1 + dim2
                        if heightindex == self.spec_data[n_file]['doppler_spectrum'].shape[1]:
                            heightindex = heightindex - 1
                            comment = comment + ' (range boundary)'
                        if timeindex == self.spec_data[n_file]['doppler_spectrum'].shape[0]:
                            timeindex = timeindex - 1
                            comment = comment + ' (time boundary)'

                        thisSpectrum = self.spec_data[n_file]['doppler_spectrum'][int(timeindex), int(heightindex), :]
                        #print(f'time index: {timeindex}, height index: {heightindex}')
                        if heightindex == -1 or timeindex == -1:
                            thisSpectrum = thisSpectrum.where(thisSpectrum.values == -999)
                            comment = comment + ' (time or range boundary)'

                        ax[dim1, dim2].plot(self.spec_data[n_file]['velocity_vectors'][c_ind, :], lin2z(thisSpectrum.values))
                        ax[dim1, dim2].set_xlim([np.nanmin(self.spec_data[n_file]['velocity_vectors'][c_ind, :]),
                                                 np.nanmax(self.spec_data[n_file]['velocity_vectors'][c_ind, :])])
                        ax[dim1, dim2].set_title(f'range:'
                                                 f'{np.round(self.spec_data[n_file]["range"].values[int(heightindex)]/1000, 2)} km,'
                                                 f' time: {format_hms(self.spec_data[n_file]["time"].values[int(timeindex)])}' + comment,
                                                 fontweight='semibold', fontsize=9, color='b')
                        # if thisnoisefloor != 0.0:
                        #    ax[dim1, dim2].axhline(h.lin2z(thisnoisefloor),color='k')
                        ax[dim1, dim2].set_xlabel("Doppler velocity [m/s]", fontweight='semibold', fontsize=9)
                        ax[dim1, dim2].set_ylabel("Reflectivity [dBZ m$^{-1}$s]", fontweight='semibold', fontsize=9)
                        #ax[dim1, dim2].set_xlim(xrange)
                        ax[dim1, dim2].grid(True)

            ax[1, 1].plot(self.spec_data[n_file]['velocity_vectors'][c_ind, :], lin2z(this_spectrum_center.values))
            ax[1, 1].set_xlabel("Doppler velocity [m/s]", fontweight='semibold', fontsize=9)
            ax[1, 1].set_ylabel("Reflectivity [dBZ m$^{-1}$s]", fontweight='semibold', fontsize=9)
            ax[1, 1].grid(True)

            ax[1, 1].set_title(f'range:'
                               f'{np.round(self.spec_data[n_file]["range"].values[int(heightindex_center)] / 1000, 2)} km,'
                               f' time: {format_hms(self.spec_data[n_file]["time"].values[int(timeindex_center)])}',
                               fontweight='semibold', fontsize=9, color='r')
            # noisefloor_center = sm.estimate_noise_hs74(thisSpectrum_center)
            # if noisefloor_center != 0.0:
            # ax[1, 1].axhline(lin2z(noisefloor_center), color='k')
            #     ax[1, 1].set_xlim(xrange)
            x = plt.ginput(self.max_peaks, timeout=0)
            # important in PyCharm:
            # uncheck Settings | Tools | Python Scientific | Show Plots in Toolwindow
            for i in range(len(x)):
                peakVals.append(x[i][0])
                peakPowers.append(x[i][1])
            plt.close()
            return peakVals, peakPowers
        else:
            return np.nan, np.nan

    def save_training_data(self):
        """
        save the marked peaks stored in TrainingData.training_data_out to a netcdf file.
        If the netcdf file does not exist yet, create it in place where spectra netcdf are stored.
        If the netcdf file does exist already, read it in, modify it and overwrite the file.
        """
        for i in range(len(self.training_data_out)):
#            r_indices = self.chirps_to_ranges(i)
#            datalist = []
#            for r in range(len(r_indices)-1):
#                datalist.append(self.training_data_out[i][:, r_indices[r]:r_indices[r+1], :])

            # create netcdf data sets if they don't exist already
            if not os.path.isfile(self.peaks_ncfiles[i]):
                data_dict = {'time': self.spec_data[i].time, 'range': self.spec_data[i].range,
                             'chirp': self.spec_data[i].chirp, 'peak': np.arange(self.max_peaks)}

                data_dict['peaks'] = (['time', 'range', 'peak'], self.training_data_out[i])
                dataset = xr.Dataset(data_dict)
                dataset.to_netcdf(self.peaks_ncfiles[i])
                print(f'created new file {self.peaks_ncfiles[i]}')

            else:
                with xr.open_dataset(self.peaks_ncfiles[i]) as data:
                    dataset = data.load()
                assert(self.training_data_out[i].shape == dataset.__getitem__('peaks').shape)
                mask = ~(self.training_data_out[i]== self.fill_value)
                dataset.__getitem__('peaks').values[mask] = self.training_data_out[i][mask]
                dataset.to_netcdf(self.peaks_ncfiles[i])
                print(f'updated file {self.peaks_ncfiles[i]}')

