# Calories_Burnt_Prediction-
This repository contains an end-to-end Machine Learning project designed to predict calories burned during physical activity based on user metrics. It features a robust data preprocessing pipeline, automated logging, and a Flask-based web interface for real-time predictions

## Project Architecture
The project is structured to separate concerns between data processing, model training, and deployment:

**main.py**: The core engine. It manages the ML lifecycle including data loading, merging, imputation, feature engineering, and model training.

**app.py**: A Flask web application that serves the trained model via a REST API to provide predictions to a frontend.

**log_code.py**: A centralized logging utility that tracks the execution flow and captures errors across different modules.

**Modular Scripts**: The pipeline relies on external modules (referenced in main.py) for specific tasks:

- **missing_value.py**: Handles data imputation.

- **variable_transform.py**: Manages outlier detection and transformations.

- **feature_selection.py**: Identifies the most impactful predictors.

- **balance_data.py**: Handles feature scaling and data balancing.

- **model.py**: Contains the Linear Regression model implementation.
