# MMA3001 CO2 Reconstruction Project

## Project Overview

This project investigates the reconstruction of missing indoor CO2 sensor measurements using numerical interpolation and machine-learning methods.

The engineering question is:

**How accurately can missing CO2 measurements be reconstructed using numerical interpolation and machine-learning methods, and how does reconstruction performance change as the duration of missing data increases?**

The project uses environmental sensor data from Monash Smart Infrastructure Dataset 2.

## Methods

Four reconstruction approaches were evaluated:

1. Linear interpolation
2. PCHIP interpolation
3. Decision Tree Regression
4. Support Vector Regression (SVR)

Artificial missing-data intervals of 10, 30, 60 and 120 minutes were created from known CO2 measurements. The original values were retained as ground truth for validation.

## Validation

Each gap duration was tested across 100 repeated locations using the same paired experimental locations for all methods.

Performance was assessed using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared
- Repeated-trial robustness analysis
- Computational runtime

## Main Finding

Linear interpolation provided the strongest overall performance for this dataset.

It produced the lowest reconstruction error, required no model training, had very low computational cost and remained robust across repeated missing-data locations.

PCHIP produced slightly higher reconstruction error, while both machine-learning methods generalised poorly to the later evaluation period despite hyperparameter tuning.

Distribution shift in several environmental predictor variables was identified as an important limitation for the machine-learning models.

## Repository Structure

```text
MMA3001-CO2-Reconstruction/
│
├── README.md
├── LICENSE
├── requirements.txt
├── notebooks/
│   └── MMA3001_CO2_Reconstruction.ipynb
├── src/
│   └── reconstruction.py
├── tests/
│   ├── test_reconstruction.py
│   └── test_report.txt
└── docs/
    ├── index.html
    ├── reconstruction.html
    └── search.js
