# SDAD
Distance-based anomaly detection of urban traffic data.

This repository contains the code for *SDAD* (Spatiotemporal Distance-based Anomaly Detection) method for urban traffic flow (count) data.

## Requirements
Implementation is done with Python 3.12.4 . 
All the required libraries with their specific versions are mentioned here:

```
pip install numpy == 1.26.4
pip install pandas == 2.2.2
pip install scikit-learn == 1.5.0
```

## Implementation
To run *SDAD* and its alternatives, follwoing steps should be taken:

- Clone this reporsitory into a local machine. 
- Download the [AnoLT](https://github.com/imaantaheri/AnoLT) dataset and copy it into the `data` folder of the cloned reporsitory.
- To run the *SDAD* itself, the code in `main.py` file should be used.
- The hyperparameters can be set at the begining of the code.
- The final performance metrics from `main.py` will be stored in the `results` folder.

To implement *SDAD-No-Cap* and *SDAD-No-Append*, the `main.py` file can again be used. 
There are two lines of code in this file that need to be removed in a specific way to implement *SDAD-No-Cap* and *SDAD-No-Append*.
instructions are provided in the code with comments around these two lines. 

For *SDAD-No-Graph* (constant universal thresholds) another python file is provided in this repository named `constant.py`. 
It applies all the possible final thresholds to anomaly scores of data and reports the best results in the end. 
The number of nearest neighbors to consider can also be specified at the begining of this code. 
The final performance metrics from `constant.py` will be stored in the `constant_results` folder.


The `extra_tools.py` also contains a number of functions being used in previously mentioned cdoes. 
Seperate execution of this file will not result in any specific output. 
