# pyPEAKO
Teresa Vogl, Martin Radenz, Heike Kalesse-Los 2022

**PEAKO** is a supervised radar Doppler spectrum peak finding algorithm. It finds the optimal 
parameters for detecting peaks in cloud radar Doppler spectra using user-generated training data. 


**PEAKO** is used to: 
- create labeled data (peaks marked by a user in cloud radar Doppler spectra), which can be used for training and testing the learned function
- train the algorithm using the labeled data to obtain the optimal parameter combination for peak detection. Optimization is done using a similarity measure based on the area below the peaks.
- test the performance of the learned function
- detect peaks in cloud radar Doppler spectra using the learned function


Reference for PEAKO: [Kalesse et al. (2019), AMT](https://www.atmos-meas-tech.net/12/4591/2019/)

Documentation is available at: [https://pypeako.readthedocs.io/en/latest/](https://pypeako.readthedocs.io/en/latest/)

-------------------

## TBD : Installation
pypeako is available via pip, so that one can simply do :
``` 
$ pip install pyPEAKO
```

## How PEAKO works
The current release is tailored to use cloud radar Doppler spectra netcdf files. The files are in the same format as the 
netcdf output files returned by the rpgpy reader ([https://github.com/actris-cloudnet/rpgpy]). 

## Contributing
If you find a bug or have ideas for improvements, or want to help develop peako, please contact one of the authors. Your
 input will for sure be appreciated! 

