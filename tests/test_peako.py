from unittest import TestCase

import pypeako
import numpy as np
import xarray as xr
import os

FILE_PATH = os.path.dirname(os.path.realpath(__file__))


class TestTrainingData(TestCase):

    input_file = f"{FILE_PATH}/../examples/sample_spectra.nc"
    training_data = pypeako.TrainingData([], num_spec=1)
    def test_add_spectrafile(self):
        self.training_data.add_spectrafile(self.input_file)

    def test_update_dimensions(self):
       self.training_data.update_dimensions()

class TestPeako(TestCase):
    input_file = f"{FILE_PATH}/../examples/marked_peaks_sample_spectra.nc"
    peako_obj = pypeako.Peako([input_file])
    def test_mask_chirps(self):
        pypeako.utils.mask_velocity_vectors(self.peako_obj.spec_data)

    def test_get_training_sample_number(self):
        self.peako_obj.create_training_mask()
        assert self.peako_obj.get_training_sample_number() == 609

    def test_train_peako(self):
        self.fail()

    def test_train_peako_inner(self):
        self.fail()

    def test_fun_to_minimize(self):
        self.fail()

    def test_area_peaks_similarity(self):
        self.fail()

    def test_assert_training(self):
        self.fail()

    def test_check_store_found_peaks(self):
        self.fail()

    def test_training_stats(self):
        self.fail()

    def test_testing_stats(self):
        self.fail()

    def test_compute_maximum_similarity(self):
        self.fail()

    def test_plot_3d_plots(self):
        self.fail()

    def test_plot_user_algorithm_spectrum(self):
        self.fail()

    def test_plot_algorithm_spectrum(self):
        self.fail()

    def test_test_peako(self):
        self.fail()

    def test_plot_numpeaks_timeheight(self):
        self.fail()

    def test_cleanup(self):
        self.fail()

