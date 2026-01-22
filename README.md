# 🔥Calories_Burnt_Prediction-
This repository contains an end-to-end Machine Learning project designed to predict calories burned during physical activity based on user metrics. It features a robust data preprocessing pipeline, automated logging, and a Flask-based web interface for real-time predictions

## ⚙️Project Architecture
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

## 🚀 Key Features
- **End-to-End Pipeline**: Automates data merging of exercise and calorie datasets, cleaning, feature engineering, and model training in one flow.

- **Real-time Inference**: A Flask-based web interface allows users to input physical metrics (Age, Heart Rate, etc.) and receive instant predictions.

- **Advanced Logging**: Utilizes a custom logging utility (log_code.py) that captures detailed execution logs and precise error line numbers for debugging.

- **Data Integrity**: Uses specialized techniques like random sample imputation and One-Hot Encoding for categorical features like 'Gender'.

🛠️ Installation & Usage
Clone the Repository:

`git clone https://github.com/your-username/calories-burnt-prediction.git`

Install Dependencies:

`pip install flask pandas numpy scikit-learn seaborn`

Run the Pipeline: Execute main.py to process data and train the model.

`python main.py`

Start the Web App: Launch the Flask server to use the prediction interface.

`python app.py`

## 📊 Model Details
The core of this project is a predictive engine designed to map physiological and activity-based metrics to caloric expenditure.

**🧠 Algorithm**: Linear Regression
The project utilizes Linear Regression as the primary model. This algorithm was chosen for its interpretability and efficiency in handling continuous numerical data, such as calculating the relationship between exercise intensity and energy output.

## 💾 Model Artifacts
For deployment in the Flask application, the following components are serialized:

calories.pkl: The trained Linear Regression model.

scaler.pkl: The fitted scaler to ensure real-time user input is transformed identically to the training data.

features.pkl: A record of the exact feature order and names required for consistent inference.
