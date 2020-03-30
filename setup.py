long_description = "Python code for detecting peaks in cloud radar Doppler spectra in a supervised machine learning " \
                   "framework"

from setuptools import setup

setup(
    name='pyPEAKO',  # pip install pyPEAKO
    version='0.0.1',
    description='peak detection in cloud radar Doppler spectra',
    long_description=long_description,
    author='Teresa Vogl',
    author_email='teresa.vogl@uni-leipzig.de',
    url='https://github.com/ti-vo/pyPEAKO',
    license='MIT',
    package_dir={'': 'peako'},
    py_modules=["mark_peaks", "train_peako"],
    package_data={},
    python_requires='>=3.6',
    install_requires=['numpy>=1.16', 'scipy>=1.2', 'netCDF4>=1.4.2',
                      'matplotlib>=3.0.2', 'xarray'],
    classifiers=[
        "Development Status :: 0 - under development",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: should be OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering"
    ]
)
