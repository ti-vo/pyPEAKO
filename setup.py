from setuptools import setup, find_packages

with open("README.md") as readme:
    long_description = readme.read()

version = {}
setup(
    name='pyPEAKO',  # pip install pypeako
    version='0.0.3.post2',
    description='peak detection in cloud radar Doppler spectra',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Teresa Vogl',
    author_email='teresa.vogl@uni-leipzig.de',
    url='https://github.com/ti-vo/pyPEAKO',
    license='MIT',
    include_package_data=True,
    packages=find_packages(),
#    package_dir={'': 'pypeako'},
#    py_modules=['pypeako'], #find_packages(exclude=['docs', 'tests', 'playground']),
#    package_data={'pyPEAKO': ['../examples/sample_spectra.nc', '../examples/marked_peaks_sample_spectra.nc']},
#    include_package_data=True,
    python_requires='>=3.6',
    install_requires=['numpy>=1.16', 'scipy>=1.2', 'netCDF4>=1.4.2',
                      'matplotlib>=3.0.2', 'xarray', 'requests', 'dask[complete]'],
    extras_require={'dev': ['pytest>=3.7', 'check-manifest', 'twine'], },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering"
    ]
)
