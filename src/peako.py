import xarray as xr
import scipy
import numpy as np
import datetime
import math
import warnings
import scipy.signal as si
import os
from scipy.optimize import differential_evolution
import random
import matplotlib
import matplotlib.pyplot as plt


def lin2z(array):
    """convert linear values to dB (for np.array or single number)"""
    return 10 * np.ma.log10(array)


def format_hms(unixtime):
    """format time stamp in seconds since 01.01.1970 00:00 UTC to HH:MM:SS"""
    return datetime.datetime.utcfromtimestamp(unixtime).strftime("%H:%M:%S")


def round_to_odd(f):
    """round to odd number"""
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
    Calculates the width (at half height) of each peak in a signal. Returns a four arrays, width, width_height,
    left and right position (edge)

    :param spectrum: 1-D ndarray, input signal
    :param pks: 1-D ndarray, indices of the peak locations (output of scipy.signal.find_peaks)
    :param left_edge: 1-D ndarray, indices of the left edges of each peak (output of functions_optimize.find_edges)
    :param right_edge: 1-D ndarray, indices of the right edges of each peak (output of functions_optimize.find_edges)
    :param rel_height: float, at which relative height compared to the peak height the width should be computed
    :return: width: array containing the width in # of Doppler bins

    """
    # initialize empty lists
    # left and right edge, position (ps) is used
    left_ps = []
    right_ps = []
    # calculate the reference height for each peak
    try:
        ref_height = spectrum[left_edge] + (spectrum[pks] - spectrum[left_edge]) * rel_height
    except IndexError:
        print(pks, left_edge, right_edge)
        raise IndexError('Likely there is an index out of bounds or empty. left edge, right edge, pks:',
                         left_edge, right_edge, pks)
    # loop over all peaks
    for i in range(len(pks)):
        # if y-value of the left peak edge is greater than the reference height, left edge is used as left position
        # I think this cannot happen by definition
        if spectrum[left_edge[i]] >= ref_height[i]:
            left_ps.append(left_edge[i])
        # else the maximum index in the interval from left edge to peak whose y-value is smaller/equal than the
        # reference height is used
        else:
            # np.where returns an array with indices from the interval which fulfill the condition starting at 1,
            # therefore the index of left edge has to be added to get the real index of the left position
            left_ps.append(max(np.where(spectrum[left_edge[i]:pks[i]] <= ref_height[i])[0]) + left_edge[i])
        # if y-value of the right peak edge is greater than the reference height, right edge is used as right
        # position
        if spectrum[right_edge[i]] >= ref_height[i]:
            right_ps.append(right_edge[i])
        # else the minimum index in the interval from peak to right edge whose y-value is smaller/equal than the
        # reference height is used
        else:
            # same as with left edge but in other direction from the peak, therefore the minimum index has to be
            # used
            # and the index of the peak has to be added
            right_ps.append(min(np.where(spectrum[pks[i]:right_edge[i] + 1] <= ref_height[i])[0]) + pks[i])

    width = [j - i for i, j in zip(left_ps,
                                   right_ps)]
    # calculate width in relation to the indices of the left and right position (edge)
    return np.asarray(width)


def find_edges(spectrum, fill_value, peak_locations):
    """
    Find the indices of left and right edges of peaks in a spectrum

    :param spectrum: a single spectrum in linear units
    :param peak_locations: indices of peaks detected for this spectrum
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
            left_edge = np.argmin(spectrum[peak_locations[p_ind - 1] : p_l])
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
    spectrum_above_noise = [max(x,0) for x in list(spectrum-noise_floor)]
    spectrum_above_noise = np.array(spectrum_above_noise)
    # Riemann sum (approximation of area):
    area = spectrum_above_noise[left_edge:right_edge].sum() * (velbins[1]-velbins[0])

    return area


def overlapping_area(edge_list_1, edge_list_2, spectrum, noise_floor, velbins):
    """ Compute maximum overlapping area of hand-marked peaks  and algorithm-detected peaks
            in a radar Doppler spectrum
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

            :param i1:
            :param i2:
            :param edge_list_1:
            :param edge_list_2:
            :param spectrum:

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

    :param velocities: list of Doppler velocities
    :param velbins: Doppler velocity bins
    :param fill_value: value to be ignored in velocities list
    :return:
    indices of closest match for each element of velocities in velbins
    """

    indices = np.asarray([argnearest(velbins, v) for v in velocities])
    indices[velocities == fill_value] = fill_value

    return indices


class Peako(object):
    def __init__(self, training_data, peak_detection='peako', optimization_method='loop',
                 smoothing_method='loess', max_peaks=5, **kwargs):

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
        :param kwargs 'training_params' = dictionary containing 't_avg', 'h_avg', 'span', 'width' and 'prom' values over
        which to loop if optimization_method is set to 'loop'.
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
        self.fill_value = self.spec_data[0]._FillValue
        self.training_params = kwargs['training_params'] if 'training params' in kwargs else \
            {'t_avg': range(2), 'h_avg': range(2), 'span': np.arange(0.005, 0.02, 0.005),
             'width': np.arange(0, 1.5, 0.5), 'prom': range(2)}
        self.training_result = {'loop': np.empty((1, 6)), 'scipy': np.empty((1, 6))}

    def create_training_mask(self):
        """
        Find the entries in Peako.training_data that have values stored in them, i.e. the indices of spectra with
        user-marked peaks. Store this mask in Peako.marked_peaks_index.
        """
        self.marked_peaks_index = []
        for f in range(len(self.training_data)):
            array_list = []
            for c in range(len(self.training_data[f].chirp)):
                var_string = f'C{c+1}peaks'
                dim_string = f'C{c+1}range'
                array_list.append(xr.DataArray(~(self.training_data[0][var_string].values[:, :, 0] == -999)*1,
                                               dims=[dim_string, 'time']))
            self.marked_peaks_index.append(array_list)

    def train_peako(self):
        """
        Train the peak finding algorithm.
        Depending on Peako.optimization_method, looping over possible parameter combinations or an optimization toolkit
        is used to find the combination of time and height averaging, smoothing span, minimum peak width and minimum
        peak prominence which yields the largest similarity between user-found and algorithm-detected peaks.

        """
        # locate the spectra that were marked by hand
        self.create_training_mask()
        # if optimization method is set to "loop", loop over possible parameter combinations:
        if self.optimization_method == 'loop':
            similarity_array = np.full([len(self.training_params[key]) for key in self.training_params.keys()], np.nan)
            for i, t_avg in enumerate(self.training_params['t_avg']):
                for j, h_avg in enumerate(self.training_params['h_avg']):
                    for k, span in enumerate(self.training_params['span']):
                        for l, wth in enumerate(self.training_params['width']):
                            for m, prom in enumerate(self.training_params['prom']):
                                peako_peaks = self.average_smooth_detect(t_avg, h_avg, span, prom, wth)
                                # compute the similarity
                                similarity = self.area_peaks_similarity(peako_peaks, array_out=False)
                                similarity_array[i, j, k, l, m] = similarity
                                self.training_result['loop'] = np.append(self.training_result['loop'],
                                                                         [[t_avg, h_avg, span, wth, prom, similarity]],
                                                                         axis=0)
                                # print(similarity, wth)

            # remove the first line from the training result
            self.training_result['loop'] = np.delete(self.training_result['loop'], 0, axis=0)

            # extract the parameter combination yielding the maximum in similarity
            t, h, s, w, p = np.unravel_index(np.argmax(similarity_array, axis=None), similarity_array.shape)
            return {'t_avg': self.training_params['t_avg'][t],
                    'h_avg': self.training_params['h_avg'][h],
                    'span': self.training_params['span'][s],
                    'width': self.training_params['width'][w],
                    'prom': self.training_params['prom'][p]}

        elif self.optimization_method == 'scipy':
            bounds = [(min(self.training_params['t_avg']), max(self.training_params['t_avg'])),
                      (min(self.training_params['h_avg']), max(self.training_params['h_avg'])),
                      (np.log10(min(self.training_params['span'])), np.log10(max(self.training_params['span']))),
                      (min(self.training_params['width']), max(self.training_params['width'])),
                      (min(self.training_params['prom']), max(self.training_params['prom']))]
            result = differential_evolution(self.fun_to_minimize, bounds=bounds)

            # remove the first line from the training result
            self.training_result['scipy'] = np.delete(self.training_result['scipy'], 0, axis=0)
            return result

    def fun_to_minimize(self, parameters):
        """
        Function which is minimized by the optimization toolkit (differential evolution).
        It averages the neighbor spectra in a range defined by t_avg and h_avg,
        calls smooth_spectrum with the defined method (Peako.smoothing_method),
        and calls get_peaks using the defined prominence and width. The t_avg, h_avg, span, width and prominence
        parameters are passed as parameters:

        :param parameters: list containing t_avg, h_avg, span, width and prominence. If this function is called within
        scipy.differential_evolution, this corresponds to the order of the elements in "bounds"
        :return: res: Result (negative similarity measure based on area below peaks); negative because optimization
        toolkits usually search for the minimum.

        """

        t_avg, h_avg, span, width, prom = parameters
        # trick to search for span in a larger (logarithmic) search space
        span = 10 ** span
        # trick to get integers:
        t_avg = np.int(round(t_avg))
        h_avg = np.int(round(h_avg))

        peako_peaks = self.average_smooth_detect(t_avg, h_avg, span, prom, width)
        # compute the similarity
        res = self.area_peaks_similarity(peako_peaks, array_out=False)

        # print(",".join(map(str, np.append(parameters, res))))
        self.training_result['scipy'] = np.append(self.training_result['scipy'],
                                                  [[t_avg, h_avg, span, width, prom, res]], axis=0)

        # for differential evolution
        # return negative similarity
        return -res

    def average_smooth_detect(self, t_avg, h_avg, span, width, prom):
        """
        Average, smooth spectra and detect peaks that fulfill prominence and with criteria

        :param t_avg: numbers of neighbors in time dimension to average over (on each side)
        :param h_avg: numbers of neighbors in range dimension to average over (on each side)
        :param span: Percentage of number of data points used for smoothing when loess or lowess smoothing is used
        :param width: minimum peak width in m/s Doppler velocity (width at half-height)
        :param prom: minimum peak prominence in dBZ

        :return:
        """
        avg_spec = self.average_spectra(t_avg, h_avg)
        smoothed_spectra = self.smooth_spectra(avg_spec, span=span)
        peaks = self.get_peaks(smoothed_spectra, prom, width)
        return peaks

    def average_spectra(self, t_avg, h_avg):
        """
        Average spectra in Peako.spec_data

        :param t_avg: number of neighboring spectra each side in time dimension to average over
        :param h_avg: number of neighboring spectra each side in range dimension to average over
        :return: avg_spec_list (list of xarray DataArrays having the same dimensions as DataArrays in Peako.spec_data,
                 containing averaged spectra)

        """
        avg_specs_list = []  # initialize empty list
        for f in range(len(self.spec_data)):
            # average spectra over neighbors in time-height
            avg_specs = self.spec_data[f].where(self.spec_data[f].time < 0).load()  # create empty xr data set
            # loop over chirps and find hand-marked spectra
            for c in range(len(self.spec_data[f].chirp)):
                h_ind, t_ind = np.where(self.marked_peaks_index[f][c] == 1)
                var_string = f'C{c+1}Zspec'

                for ind in range(len(h_ind)):
                    # This will throw a RuntimeWarning : Mean of empty slice which can be annoying, so we're ignoring
                    # warnings here.
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        avg_specs[var_string][h_ind[ind], t_ind[ind], :] = np.nanmean(
                            self.spec_data[f][var_string].values[(h_ind[ind] - h_avg): (h_ind[ind] + h_avg + 1),
                                                                 (t_ind[ind] - t_avg): (t_ind[ind] + t_avg + 1),
                                                                 :], axis=(0, 1))
            avg_specs_list.append(avg_specs)

        return avg_specs_list

    def get_peaks(self, spectra, prom, width_thresh):
        """
        detect peaks in (smoothed) spectra which fulfill minimum prominence and width criteria.

        :param spectra: list of data arrays containing (smoothed) spectra in linear units
        :param prom: minimum prominence in dbZ
        :param width_thresh: width threshold in m/s
        :return: peaks: list of data arrays containing detected peak indices. Length of this list is the same as the
        length of the spectra (input parameter) list.
        """
        peaks = []
        for f in range(len(spectra)):
            peaks_dataset = xr.Dataset()
            for c in range(len(spectra[f].chirp)):
                h_ind, t_ind = np.where(self.marked_peaks_index[f][c] == 1)
                peaks_array = xr.Dataset(data_vars={f'C{c+1}PeakoPeaks': xr.DataArray(np.full((spectra[f][f'C{c+1}Zspec'].values.shape[0:2] + (self.max_peaks,)),
                                         np.nan, dtype=np.int),
                                         dims=[f'C{c+1}range', 'time', 'peaks'])})
                # convert width_thresh units from m/s to # of Doppler bins:
                width_thresh = width_thresh/np.nanmedian(np.diff(self.spec_data[f][f'C{c+1}vel'].values))
                for h, t in zip(h_ind, t_ind):
                    # extract one spectrum at a certain height/ time, convert to dBZ and fill masked values with minimum
                    # of spectrum
                    spectrum = spectra[f][f'C{c + 1}Zspec'].values[h, t, :]
                    spectrum = lin2z(spectrum)
                    spectrum.data[spectrum.mask] = np.nanmin(spectrum)
                    spectrum = spectrum.data
                    # call scipy.signal.find_peaks to detect peaks in the (logarithmic) spectrum
                    # it is important that nan values are not included in the spectrum passed to si
                    locs, props = si.find_peaks(spectrum, prominence=prom)
                    # find left and right edges of peaks
                    le, re = find_edges(spectra[f][f'C{c+1}Zspec'].values[h, t, :], self.fill_value, locs)
                    # compute the width
                    width = peak_width(spectra[f][f'C{c+1}Zspec'].values[h, t, :], locs, le, re)
                    locs = locs[width > width_thresh]
                    locs = locs[0: self.max_peaks] if len(locs) > self.max_peaks else locs
                    peaks_array[f'C{c+1}PeakoPeaks'].values[h, t, 0:len(locs)] = locs
                peaks_dataset.update(other=peaks_array)
            peaks.append(peaks_dataset)

        return peaks

    def find_peaks_peako(self, t_avg, h_avg, span, prom, wth):
        # call average_smooth_detect
        pass

    def find_peaks_peaktree(self, prom):
        pass

    def smooth_spectra(self, spectra, span):
        """
        smooth an array of spectra. 'loess' and 'lowess' methods apply a Savitzky-Golay filter to an array.
        Refer to scipy.signal.savgol_filter for documentation on the 1-d filter. 'loess' means that polynomial is
        degree 2; lowess means polynomial is degree 1.

        :param spectra: list of Datasets of spectra
        :param span: span used for loess/ lowess smoothing
        :param velbins: Doppler velocity bins in m/s
        :return: spectra_out, an array with same dimensions as spectra containing the smoothed spectra
        """
        method = self.smoothing_method
        spectra_out = [i.copy(deep=True) for i in spectra]
        for f in range(len(spectra)):
            for c in range(len(spectra[f].chirp)):
                var_string = f'C{c+1}vel'
                velbins = self.spec_data[f][var_string].values
                window_length = round_to_odd(span * len(velbins))
                if method == 'loess':
                    spectra_out[f][f'C{c+1}Zspec'].values = scipy.signal.savgol_filter(spectra[f][f'C{c+1}Zspec'].values,
                                                                                       window_length,
                                                                                       polyorder=2, axis=2,
                                                                                       mode='nearest')
                elif method == 'lowess':
                    spectra_out[f][f'C{c + 1}Zspec'].values = scipy.signal.savgol_filter(
                        spectra[f][f'C{c + 1}Zspec'].values,
                        window_length,
                        polyorder=1, axis=2,
                        mode='nearest')
        return spectra_out

    def area_peaks_similarity(self, algorithm_peaks, array_out=False):
        """ Compute similarity measure based on overlapping area of hand-marked peaks by a user and algorithm-detected
            peaks in a radar Doppler spectrum

            :param algorithm_peaks: ndarray of indices of spectrum where peako detected peaks
            :param array_out: Bool. If True, area_peaks_similarity will return a list of xr.Datasets containing the
            computed similarities for each spectrum in the time-height grid. If False, the integrated similarity (sum)
            of all the hand-marked spectra is returned. Default is False.

        """
        sim_out = [] if array_out else 0
        # TODO add output data structure for array_out = True

        # loop over files and chirps, and then over the spectra which were marked by hand
        for f in range(len(self.specfiles)):
            for c in range(len(self.training_data[f].chirp)):
                velbins = self.spec_data[f][f'C{c+1}vel'].values
                h_ind, t_ind = np.where(self.marked_peaks_index[f][c] == 1)
                for h, t in zip(h_ind, t_ind):
                    user_peaks = self.training_data[f][f'C{c+1}peaks'].values[h, t, :]
                    user_peaks = user_peaks[~(user_peaks == self.fill_value)]
                    # convert velocities to indices
                    user_peaks = np.asarray([argnearest(velbins, val) for val in user_peaks])
                    spectrum = self.spec_data[f][f'C{c+1}Zspec'].values[h, t, :]
                    user_peaks.sort()
                    peako_peaks = algorithm_peaks[f][f'C{c+1}PeakoPeaks'].values[h, t, :]
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
                                                                           spectrum, np.nanmin(spectrum), velbins)
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
                        similarity = similarity - area_above_floor(le_alg_peaks[i], re_alg_peaks[i], spectrum,
                                                                   np.nanmin(spectrum), velbins)
                    for i in range(len(le_user_peaks)):
                        similarity = similarity - area_above_floor(le_user_peaks[i], re_user_peaks[i], spectrum,
                                                                   np.nanmin(spectrum), velbins)
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
        assert(len(self.marked_peaks_index) > 0), "no training mask available"
        assert(self.training_result['loop'].shape[0] + self.training_result['scipy'].shape[0] > 2), "no training result"

    def training_stats(self, make_3d_plots=False):
        """
        print out training statistics
        :param make_3d_plots: bool: Default is False. If set to True, plot_3d_plots will be called
        """

        self.assert_training()
        # compute maximum possible similarity
        user_peaks = []
        for f in range(len(self.specfiles)):
            peaks_dataset = xr.Dataset()
            for c in range(len(self.training_data[f].chirp)):
                velbins = self.spec_data[f][f'C{c+1}vel'].values
                peaks_array = xr.Dataset(data_vars={f'C{c+1}PeakoPeaks': xr.DataArray(np.full(
                    self.training_data[f][f'C{c+1}peaks'].values.shape,
                    np.nan, dtype=np.int), dims=[f'C{c+1}range', 'time', 'peaks'])})
                # convert m/s to indices (call vel_to_ind)
                h_ind, t_ind = np.where(self.marked_peaks_index[f][c] == 1)
                for h, t in zip(h_ind, t_ind):
                    indices = vel_to_ind(self.training_data[f][f'C{c + 1}peaks'].values[h, t, :], velbins,
                                         self.fill_value)
                    peaks_array[f'C{c+1}PeakoPeaks'].values[h, t, :] = indices
                peaks_dataset.update(other=peaks_array)
            user_peaks.append(peaks_dataset)

        maximum_similarity = self.area_peaks_similarity(user_peaks)
        for j in self.training_result.keys():
            if self.training_result[j].shape[0] > 1:
                print(f'{j}:')
                catch = np.nanmax(self.training_result[j][:, -1])
                print(f'similarity is {round(catch/maximum_similarity*100,2)}% of maximum possible similarity')
                print('h_avg: {0[0]}, t_avg:{0[1]}, span:{0[2]}, width: {0[3]}, prom: {0[4]}'.format(
                    (self.training_result[j][np.argmax(self.training_result[j][:, -1]), :-1])))

                if make_3d_plots:
                    fig, ax = self.plot_3d_plots(j)
        return maximum_similarity

    def plot_3d_plots(self, key):
        """
        Generates 4 panels of 3D plots of parameter vs. parameter vs. similarity for evaluating the training of pyPEAKO
        by eye

        :param key: dictionary key in Peako.training_result for which to make the 3D plots, either 'loop' or 'scipy'.
        :return: fig, ax : matplotlib.pyplot figure and axes
        """

        from mpl_toolkits.mplot3d import Axes3D

        training_result = self.training_result[key]
        fig, ax = plt.subplots(2, 2, subplot_kw=dict(projection='3d'))
        ax[0, 0].scatter(training_result[:,0], training_result[:,1], training_result[:,-1], zdir='z',
                         c=training_result[:, -1],
                   cmap='seismic', vmin=-np.nanmax(training_result[:, -1]), vmax=np.nanmax(training_result[:, -1]))
        ax[0, 0].set_xlabel('height averages')
        ax[0, 0].set_ylabel('time averages')
        ax[0, 0].set_zlabel('similarity')

        ax[1, 1].scatter(training_result[:,3], training_result[:,2], training_result[:,-1], zdir='z',
                         c=training_result[:, -1],
                   cmap='seismic', vmin=-np.nanmax(training_result[:, -1]), vmax=np.nanmax(training_result[:, -1]))
        ax[1, 1].set_xlabel('width')
        ax[1, 1].set_ylabel('span')
        ax[1, 1].set_zlabel('similarity')

        ax[0, 1].scatter(training_result[:,4], training_result[:,3], training_result[:,-1], zdir='z',
                         c=training_result[:, -1],
                   cmap='seismic', vmin=-np.nanmax(training_result[:, -1]), vmax=np.nanmax(training_result[:, -1]))
        ax[0, 1].set_xlabel('prom')
        ax[0, 1].set_ylabel('width')
        ax[0, 1].set_zlabel('similarity')

        ax[1, 0].scatter(training_result[:, 4], training_result[:, 1], training_result[:, -1], zdir='z',
                         c=training_result[:, -1],
                   cmap='seismic', vmin=-np.nanmax(training_result[:, -1]), vmax=np.nanmax(training_result[:, -1]))
        ax[1, 0].set_xlabel('prom')
        ax[1, 0].set_ylabel('time averages')
        ax[1, 0].set_zlabel('similarity')

        return fig, ax

    def plot_user_algorithm_spectrum(self, **kwargs):

        # assert that training has happened:
        self.assert_training()

        # set random seed if it's given in the key word arguments:
        if 'seed' in kwargs:
            random.seed(kwargs['seed'])

        # select a random user-marked spectrum
        f = random.randint(0, len(self.marked_peaks_index) - 1)
        c = random.randint(0, len(self.marked_peaks_index[f]) - 1)
        h_ind, t_ind = np.where(self.marked_peaks_index[f][c] == 1)
        i = random.randint(0, len(h_ind) - 1)
        velbins = self.spec_data[f][f'C{c+1}vel'].values
        spectrum = self.spec_data[f][f'C{c+1}Zspec'].values[h_ind[i], t_ind[i], :]
        user_ind = vel_to_ind(self.training_data[f][f'C{c+1}peaks'].values[h_ind[i], t_ind[i], :], velbins=velbins,
                              fill_value=self.fill_value)
        user_ind = user_ind[user_ind > 0]


        # TODO move part below into a separate function and store DataArray in the Peako object
        peako_peaks = {}
        for j in self.training_result.keys():
            if self.training_result[j].shape[0] > 1:
                i_max = np.argmax(self.training_result[j][:, -1])
                h, t, s, w, p = self.training_result[j][i_max, :-1]
                peako_peaks[j] = self.average_smooth_detect(t_avg=int(t), h_avg=int(h), span=s, width=w, prom=p)

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
        for j in peako_peaks.keys():
            peako_ind = peako_peaks[j][f][f'C{c + 1}PeakoPeaks'].values[h_ind[i], t_ind[i], :]
            peako_ind = peako_ind[peako_ind > 0]

            ax.plot(velbins[peako_ind], lin2z(spectrum)[peako_ind], marker='o',
                    color=['#0339cc', '#0099ff', '#9933ff'][c_ind], markeredgecolor='k',
                linestyle="None", label=f'PEAKO peaks {j}', markersize=8)
            c_ind += 1

        ax.plot(velbins[user_ind], lin2z(spectrum)[user_ind], marker=cut_star, color='r',
                linestyle="None", label='user peaks')
        ax.set_xlabel('Doppler Velocity [m s$^{-1}$]', fontweight='semibold', fontsize=fsz)
        ax.set_ylabel('Reflectivity [dBZ m$\\mathregular{^{-1}}$ s]', fontweight='semibold', fontsize=fsz)
        ax.grid(linestyle=':')
        ax.set_xlim(np.nanmin(velbins), np.nanmax(velbins))
        ax.legend(fontsize=fsz)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        ax.set_title(f'spectrum at {round(self.spec_data[f][f"C{c+1}range"].values[h_ind[i]])} m, '
                     f'{format_hms(self.spec_data[f]["time"].values[int(t_ind[i])])}')

        return fig, ax


class TrainingData(object):
    def __init__(self, specfiles_in, num_spec=[30], max_peaks=5):
        """
        :param specfiles_in: list of radar spectra files (netcdf format)
        :param num_spec: (list) number of spectra to mark by the user (default 30)
        :param maxpeaks: (int) maximum number of peaks per spectrum (default 5)

        """
        self.specfiles_in = specfiles_in
        self.spec_data = [xr.open_mfdataset(fin, combine='by_coords', ) for fin in specfiles_in]
        self.num_spec = []
        self.tdim = []
        self.rdim = []
        self.training_data_out = []
        self.peaks_ncfiles = []
        self.plot_count = []

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
            self.rdim.append(self.chirps_to_ranges(f)[-1])
            self.training_data_out.append(np.full((self.rdim[-1], self.tdim[-1], self.max_peaks), np.float64(-999)))
            ncfile = '/'.join(self.specfiles_in[f].split('/')[0:-1]) + \
                     '/' + 'marked_peaks_' + self.specfiles_in[f].split('/')[-1]
            self.peaks_ncfiles.append(ncfile)
            self.plot_count.append(0)

    def chirps_to_ranges(self, f):
        """
        extract a list of range indices from the range variables of all chirps
        :param f:  index of the file in TrainingData.spec_data
        :return: a list [0, len(C1Range), len(C1Range) + len(C2Range), ...]
        """
        n_chirps = len(self.spec_data[f]['chirp'])
        n_rg = [0]
        # loop over chirps
        for c in range(n_chirps):
            # name of range variable is e.g. C1range, add lengths of this variable for all chirps
            var_string = f'C{c + 1}range'
            n_rg.append(len(self.spec_data[f][var_string]) + n_rg[-1])
        return n_rg

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

                vals, powers = self.input_peak_locations(n, random_index_t, random_index_r)
                if not np.all(np.isnan(vals)):
                    self.training_data_out[n][random_index_r, random_index_t, 0:len(vals)] = vals
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
        n_rg = self.chirps_to_ranges(n_file)
        c_ind = np.digitize(r_index, n_rg)
        #print(f'range index {r_index} is in chirp {c_ind} with ranges in chirps {n_rg[1:]}')

        heightindex_center = r_index - n_rg[c_ind - 1]
        timeindex_center = t_index
        this_spectrum_center = self.spec_data[n_file][f'C{c_ind}Zspec'][int(heightindex_center), int(timeindex_center), :]

        #print(f'time index center: {timeindex_center}, height index center: {heightindex_center}')
        if not np.sum(~(this_spectrum_center.values == -999)) == 0:
            # if this spectrum is not empty, we plot 3x3 panels with shared x and y axes
            fig, ax = plt.subplots(3, 3, figsize=[11, 11], sharex=True, sharey=True)
            fig.suptitle(f'Mark peaks in spectrum in center panel. Fig. {self.plot_count[n_file]+1} out of '
                         f'{self.num_spec[n_file]}; File {n_file+1} of {len(self.spec_data)}')
            for dim1 in range(3):
                for dim2 in range(3):
                    if not (dim1 == 1 and dim2 == 1):  # if this is not the center panel plot
                        comment = ''
                        heightindex = r_index - n_rg[c_ind-1] + 1 - dim1
                        timeindex = t_index - 1 + dim2
                        if heightindex == self.spec_data[n_file][f'C{c_ind}Zspec'].shape[0]:
                            heightindex = heightindex - 1
                            comment = comment + ' (range boundary)'
                        if timeindex == self.spec_data[n_file][f'C{c_ind}Zspec'].shape[1]:
                            timeindex = timeindex - 1
                            comment = comment + ' (time boundary)'

                        thisSpectrum = self.spec_data[n_file][f'C{c_ind}Zspec'][int(heightindex), int(timeindex), :]
                        #print(f'time index: {timeindex}, height index: {heightindex}')
                        if heightindex == -1 or timeindex == -1:
                            thisSpectrum = thisSpectrum.where(thisSpectrum.values==-999)
                            comment = comment + ' (time or range boundary)'

                        ax[dim1, dim2].plot(self.spec_data[n_file][f'C{c_ind}vel'], lin2z(thisSpectrum.values))
                        ax[dim1, dim2].set_xlim([np.nanmin(self.spec_data[n_file][f'C{c_ind}vel']),
                                                 np.nanmax(self.spec_data[n_file][f'C{c_ind}vel'])])
                        ax[dim1, dim2].set_title(f'range:'
                                                 f'{np.round(self.spec_data[n_file][f"C{c_ind}range"].values[int(heightindex)]/1000, 2)} km,'
                                                 f' time: {format_hms(self.spec_data[n_file]["time"].values[int(timeindex)])}' + comment,
                                                 fontweight='semibold', fontsize=9, color='b')
                        # if thisnoisefloor != 0.0:
                        #    ax[dim1, dim2].axhline(h.lin2z(thisnoisefloor),color='k')
                        ax[dim1, dim2].set_xlabel("Doppler velocity [m/s]", fontweight='semibold', fontsize=9)
                        ax[dim1, dim2].set_ylabel("Reflectivity [dBZ m$^{-1}$s]", fontweight='semibold', fontsize=9)
                        #ax[dim1, dim2].set_xlim(xrange)
                        ax[dim1, dim2].grid(True)

            ax[1, 1].plot(self.spec_data[n_file][f'C{c_ind}vel'], lin2z(this_spectrum_center.values))
            ax[1, 1].set_xlabel("Doppler velocity [m/s]", fontweight='semibold', fontsize=9)
            ax[1, 1].set_ylabel("Reflectivity [dBZ m$^{-1}$s]", fontweight='semibold', fontsize=9)
            ax[1, 1].grid(True)

            ax[1, 1].set_title(f'range:'
                               f'{np.round(self.spec_data[n_file][f"C{c_ind}range"].values[int(heightindex_center)] / 1000, 2)} km,'
                               f' time: {format_hms(self.spec_data[n_file]["time"].values[int(timeindex_center)])}',
                               fontweight='semibold', fontsize=9, color='r')
            # noisefloor_center = sm.estimate_noise_hs74(thisSpectrum_center)
            # if noisefloor_center != 0.0:
            # ax[1, 1].axhline(lin2z(noisefloor_center), color='k')
            #     ax[1, 1].set_xlim(xrange)
            x = plt.ginput(self.max_peaks, timeout=0)
            # important in PyCharm
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
        # bring TrainingData.training_data_out into the correct format
        for i in range(len(self.training_data_out)):
            r_indices = self.chirps_to_ranges(i)
            datalist = []
            for r in range(len(r_indices)-1):
                datalist.append(self.training_data_out[i][r_indices[r]:r_indices[r+1], :, :])

            # create netcdf data sets if they don't exist already
            if not os.path.isfile(self.peaks_ncfiles[i]):
                data_dict = {'time': self.spec_data[i].time, 'chirp': self.spec_data[i].chirp}
                for r in range(len(r_indices)-1):
                    key = f'C{r+1}range'
                    data_dict[key] = self.spec_data[i].__getitem__(key)
                    key_2 = f'C{r+1}peaks'
                    data_dict[key_2] = ([key, 'time', 'peaks'], datalist[r])
                dataset = xr.Dataset(data_dict)
                dataset.to_netcdf(self.peaks_ncfiles[i])
                print(f'created new file {self.peaks_ncfiles[i]}')

            else:
                with xr.open_dataset(self.peaks_ncfiles[i]) as data:
                    dataset = data.load()
                for r in range(len(r_indices)-1):
                    key = f'C{r+1}peaks'
                    assert(datalist[r].shape == dataset.__getitem__(key).shape)
                    mask = ~(datalist[r] == -999)
                    dataset.__getitem__(key).values[mask] = datalist[r][mask]
                    dataset.to_netcdf(self.peaks_ncfiles[i])
                    print(f'updated file {self.peaks_ncfiles[i]}')

