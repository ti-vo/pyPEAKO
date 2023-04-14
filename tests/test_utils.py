from unittest import TestCase
import pypeako.utils as utils

class Test(TestCase):
    def test_lin2z(self):
        assert utils.lin2z(100) == 20


class Test(TestCase):
    def test_format_hms(self):
        self.fail()
