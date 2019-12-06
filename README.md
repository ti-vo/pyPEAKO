# pyPEAKO


**PEAKO** is a supervised radar Doppler spectrum peak finding algorithm. It finds the optimal 
parameters for detecting peaks in cloud radar Doppler spectra using user-generated training data. 
PEAKO can be used to find the best parameters for peak detection in the peakTree algorithm, which is currently 
adapted to run in CLOUDNET.

**PEAKO** is used to: 
- create labeled data (peaks marked by a user in Radar Doppler spectra), which can be used for training and testing the learned function
- train the algorithm using the labeled data to obtain the optimal parameter combination for peak detection based on a similarity measure based on the area below the peaks
- test the performance of the learned function [TBD]
- detect peaks in radar Doppler spectra using the learned function


Reference for PEAKO: [Kalesse et al. (2019), AMT](https://www.atmos-meas-tech.net/12/4591/2019/)

<img src="doc/example_spectrum.png">

Installation
-------------------

Clone the repository, e.g. by
 ```
$ git clone https://github.com/ti-vo/pyPEAKO
```

navigate to the main folder (pyPEAKO):

```
$ python3 setup.py install --user
```


