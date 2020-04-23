

from setuptools import setup, find_packages

with open("README.md") as readme:
    long_description = readme.read()

setup(
    name='pyPEAKO',  # pip install pypeako
    version='0.0.2',
    description='peak detection in cloud radar Doppler spectra',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Teresa Vogl',
    author_email='teresa.vogl@uni-leipzig.de',
    url='https://github.com/ti-vo/pyPEAKO',
    license='MIT',
    package_dir={'': 'src'},
    py_modules=['peako'], #find_packages(exclude=['docs', 'tests', 'playground']),
    package_data={'src': ['sample_spectra.nc', 'marked_peaks_sample_spectra.nc']},
    python_requires='>=3.6',
    install_requires=['numpy>=1.16', 'scipy>=1.2', 'netCDF4>=1.4.2',
                      'matplotlib>=3.0.2', 'xarray'],
    extras_require={'dev': ['pytest>=3.7', 'check-manifest', 'twine'], },
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering"
    ]
)
