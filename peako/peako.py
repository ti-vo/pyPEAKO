import xarray as xr
import scipy
import numpy as np
import datetime
import scipy.signal as si
from scipy.optimize import differential_evolution


def lin2z(array):
    """convert linear values to dB (for np.array or single number)"""
    return 10 * np.ma.log10(array)


def format_hms(unixtime):
    """format time stamp in seconds since 01.01.1970 00:00 UTC to HH:MM:SS"""
    return datetime.datetime.utcfromtimestamp(unixtime).strftime("%H:%M:%S")


def round_to_odd(f):
    """round to odd number"""
    return int(np.ceil(f / 2.) * 2 + 1)


class Peako(object):
    def __init__(self, training_data, peak_detection='peako', optimization_method='loop',
                 smoothing_method='loess'):
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
        """
        self.training_files = training_data
        self.training_data = [xr.open_mfdataset(fin, combine='by_coords') for fin in training_data]
        self.specfiles = ['/'.join(f.split('/')[:-1]) + '/' + f.split('/')[-1][13:] for f in self.training_files]
        self.spec_data = [xr.open_mfdataset(fin, combine='by_coords') for fin in self.specfiles]
        self.peak_detection_method = peak_detection
        self.optimization_method = optimization_method
        self.smoothing_method = smoothing_method
        self.marked_peaks_index = []

    def create_training_mask(self):
        """
        Find the entries in Peako.training_data that have values stored in them, i.e. the indices of spectra with
        user-marked peaks. Store this mask in Peako.marked_peaks_index.
        """
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
        Train the peako algorithm.
        """
        # locate the spectra that were marked by hand
        self.create_training_mask()
        # if optimization method is set to "loop", loop over possible parameter combinations
        if self.optimization_method == 'loop':
            for t_avg in range(2):
                for h_avg in range(2):
                    for span in np.arange(0.005, 0.02, 0.005):
                        for wth in np.arange(0, 1.5, 0.5):
                            for prom in range(2):
                                self.average_smooth_detect(t_avg, h_avg, span, prom, wth)

    def average_smooth_detect(self, t_avg, h_avg, span, prom, width):
        """
        Average, smooth spectra and detect peaks that fulfill prominence and with criteria
        :param t_avg: numbers of neighbors in time dimension to average over (on each side)
        :param h_avg: numbers of neighbors in range dimension to average over (on each side)
        :param span: Percentage of number of data points used for smoothing when loess or lowess smoothing is used
        :param prom: minimum peak prominence in dBZ
        :param width: minimum peak width in m/s Doppler velocity (width at half-height)
        :return:
        """
        avg_spec = self.average_spectra(t_avg, h_avg)
        smoothed_spectra = self.smooth_spectra(avg_spec, span=span)
        peaks = self.get_peaks(lin2z(smoothed_spectra), prom, width)

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
                    avg_specs[var_string][h_ind[ind], t_ind[ind], :] = np.nanmean(self.spec_data[f][var_string].values[
                                                                       (h_ind[ind] - h_avg): (h_ind[ind] + h_avg + 1),
                                                                       (t_ind[ind] - t_avg): (t_ind[ind] + t_avg + 1),
                                                                       :], axis=(0, 1))
            avg_specs_list.append(avg_specs)

        return avg_specs_list

    def get_peaks(self, spectra, prom, width_thresh):
        locs, props = si.find_peaks(spectra, prominence=prom)
        le, re = self.find_edges(spectra, locs)
        width = self.peak_width(spectra, locs, le, re)
        locs = locs[width > width_thresh]

        return locs

    def find_peaks_peako(self, t_avg, h_avg, span, prom, wth):
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

    def fun_to_minimize(self):
        pass


    def area_peaks_similarity(self, spectrum, user_peaks, algorithm_peaks, velbins):
        """ Compute similarity measure based on overlapping area of hand-marked peaks by a user and algorithm-detected peaks
            in a radar Doppler spectrum

            :param spectrum: ndarray containing reflectivity in dB units, contains nan values
            :param user_peaks: ndarray of indices of spectrum where user marked peaks
            :param algorithm_peaks: ndarray of indices of spectrum where peako detected peaks
            :param velbins: ndarray of same length as spectrum, from -Nyquist to +Nyquist Doppler velocity (m/s)

        """
        user_peaks.sort()
        algorithm_peaks.sort()
        le_user_peaks, re_user_peaks = find_edges(spectrum, user_peaks)
        le_alg_peaks, re_alg_peaks = find_edges(spectrum, algorithm_peaks)

        similarity = 0
        overlap_area = math.inf
        while((len(algorithm_peaks) > 0) & (len(user_peaks) >0) & (overlap_area >0)):
            # compute maximum overlapping area
            user_ind, alg_ind, overlap_area = overlapping_area([le_user_peaks, re_user_peaks], [le_alg_peaks, re_alg_peaks],
                                                               spectrum, np.nanmin(spectrum), velbins)
            similarity = similarity + overlap_area
            if not user_ind is None:
                user_peaks = np.delete(user_peaks, user_ind)
                le_user_peaks = np.delete(le_user_peaks, user_ind)
                re_user_peaks = np.delete(re_user_peaks, user_ind)
            if not alg_ind is None:
                algorithm_peaks = np.delete(algorithm_peaks, alg_ind)
                le_alg_peaks = np.delete(le_alg_peaks, alg_ind)
                re_alg_peaks = np.delete(re_alg_peaks, alg_ind)

        # Subtract area of non-overlapping regions
        for i in range(len(le_alg_peaks)):
            similarity = similarity - area_above_floor(le_alg_peaks[i], re_alg_peaks[i], spectrum, np.nanmin(spectrum), velbins)
        for i in range(len(le_user_peaks)):
            similarity = similarity - area_above_floor(le_user_peaks[i], re_user_peaks[i], spectrum, np.nanmin(spectrum), velbins)
        #print(user_peaks, algorithm_peaks, similarity)


        return similarity




# for testing
if __name__ == '__main__':
    P = Peako(['/home/tvogl/PhD/radar_data/W_band_Punta/cloudnet_format/marked_peaks_20190222-1916-1925_LIMRAD94_spectra.nc'])
    P.create_training_mask()
    a = P.average_spectra(0, 1)
    b = P.smooth_spectra(a, 0.05)
    import matplotlib
    matplotlib.use('TkAgg')

    import matplotlib.pyplot as plt
    hind, tind = np.where(P.marked_peaks_index[0][0] == 1)
    fig, ax = plt.subplots(1)
    ax.plot(P.spec_data[0].C1velocity.values, lin2z(P.spec_data[0].C1Zspec[hind[1], tind[1]]), label='raw')
    ax.plot(a[0].C1velocity.values, lin2z(a[0].C1Zspec[hind[1], tind[1]]), label='averaged')
    ax.plot(b[0].C1velocity.values, lin2z(b[0].C1Zspec[hind[1], tind[1]]), label='averaged, smoothed')
    ax.legend()
    P.train_peako()