import xarray as xr
import random
import numpy as np
import os
import peako
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


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

        matplotlib.use('TkAgg')
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

                        ax[dim1, dim2].plot(self.spec_data[n_file][f'C{c_ind}vel'], peako.lin2z(thisSpectrum.values))
                        ax[dim1, dim2].set_xlim([np.nanmin(self.spec_data[n_file][f'C{c_ind}vel']),
                                                 np.nanmax(self.spec_data[n_file][f'C{c_ind}vel'])])
                        ax[dim1, dim2].set_title(f'range:'
                                                 f'{np.round(self.spec_data[n_file][f"C{c_ind}range"].values[int(heightindex)]/1000, 2)} km,'
                                                 f' time: {peako.format_hms(self.spec_data[n_file]["time"].values[int(timeindex)])}' + comment,
                                                 fontweight='semibold', fontsize=9, color='b')
                        # if thisnoisefloor != 0.0:
                        #    ax[dim1, dim2].axhline(h.lin2z(thisnoisefloor),color='k')
                        ax[dim1, dim2].set_xlabel("Doppler velocity [m/s]", fontweight='semibold', fontsize=9)
                        ax[dim1, dim2].set_ylabel("Reflectivity [dBZ m$^{-1}$s]", fontweight='semibold', fontsize=9)
                        #ax[dim1, dim2].set_xlim(xrange)
                        ax[dim1, dim2].grid(True)

            ax[1, 1].plot(self.spec_data[n_file][f'C{c_ind}vel'], peako.lin2z(this_spectrum_center.values))
            ax[1, 1].set_xlabel("Doppler velocity [m/s]", fontweight='semibold', fontsize=9)
            ax[1, 1].set_ylabel("Reflectivity [dBZ m$^{-1}$s]", fontweight='semibold', fontsize=9)
            ax[1, 1].grid(True)

            ax[1, 1].set_title(f'range:'
                                     f'{np.round(self.spec_data[n_file][f"C{c_ind}range"].values[int(heightindex_center)] / 1000, 2)} km,'
                                     f' time: {peako.format_hms(self.spec_data[n_file]["time"].values[int(timeindex_center)])}',
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


# for testing:
dummy = TrainingData(['/home/tvogl/PhD/radar_data/W_band_Punta/cloudnet_format/20190222-1916-1925_LIMRAD94_spectra.nc',
                      '/home/tvogl/PhD/radar_data/W_band_Punta/cloudnet_format/20190222-1616-1625_LIMRAD94_spectra.nc'],
                     num_spec=[13, 13])
dummy.mark_random_spectra()
dummy.save_training_data()
