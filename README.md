# SDAD
Distance-based anomaly detection of urban traffic data.

This repository contains the code for * *SDAD (Spatiotemporal Distance-based Anomaly Detection) method for urban traffic flow (count) data.

## Requirements
Implementation is done with Python 3.12.4 . 
All the required libraries with their specific versions are mentioned here:

```
pip install numpy == 1.26.4
pip install pandas == 2.2.2
pip install scikit-learn == 1.5.0
```

## Implementation
To run * *SDAD and its alternatives follwoing steps should be taken:

- Clone this reporsitory into a local machine. 
- Download the [AnoLT](https://github.com/imaantaheri/AnoLT) dataset and copy it into the `data` folder of the cloned reporsitory.
- To run the SDAD itself, the code in `main.py` file should be used.
- The hyperparameters can be set at the begining of the code.
- The final performance metrics from `main.py` will be stored in the `results` folder.

To implement * *SDAD-no
