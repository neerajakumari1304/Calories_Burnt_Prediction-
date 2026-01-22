import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

import warnings
warnings.filterwarnings('ignore')

from log_code import setup_logging
logger = setup_logging('model')

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from sklearn.linear_model import Ridge, Lasso
import pickle

class LR_MODEL:

    def linear_regression(X_train, y_train, X_test, y_test):
        try:
            with open("features.pkl", "wb") as f:
                pickle.dump([col for col in X_train.columns if col != 'index'], f)
            #Linear Regression
            logger.info('Linear Regression')
            reg = LinearRegression()
            reg.fit(X_train, y_train)
            logger.info(f'Intercept : {reg.intercept_}')
            logger.info(f'Coefficient : {reg.coef_}')
            y_train_pred_LR = reg.predict(X_train)
            y_test_pred_LR = reg.predict(X_test)
            c = pd.DataFrame({'y_train': y_train, 'y_train_pred_LR': y_train_pred_LR})
            logger.info(f'{c.sample(10)}')
            logger.info(f'Training Accuracy (r2_score) Using Linear Regression : {r2_score(y_train, y_train_pred_LR)}')
            logger.info(f'Training Loss (Mean_squared_error) Using Linear Regression : {mean_squared_error(y_train, y_train_pred_LR)}')
            logger.info(f'Test Accuracy (r2_score) Using Linear Regression : {r2_score(y_test, y_test_pred_LR)}')
            logger.info(f'Test Loss (Mean_squared_error) Using Linear Regression : {mean_squared_error(y_test, y_test_pred_LR)}')
            '''
            #Ridge Regression
            logger.info('Ridge Regression')
            reg_ridge = Ridge(alpha=0.5)
            reg_ridge.fit(X_train, y_train)
            logger.info(f'Intercept : {reg_ridge.intercept_}')
            logger.info(f'Coefficient : {reg_ridge.coef_}')
            y_train_pred_Ridge = reg_ridge.predict(X_train)
            y_test_pred_Ridge = reg_ridge.predict(X_test)
            c ['y_train_pred_Ridge'] =  y_train_pred_Ridge
            logger.info(f'{c.sample(10)}')
            logger.info(f'Training Accuracy (r2_score) Using Ridge: {r2_score(y_train, y_train_pred_Ridge)}')
            logger.info(f'Training Loss (Mean_squared_error) Using Ridge: {mean_squared_error(y_train, y_train_pred_Ridge)}')
            logger.info(f'Test Accuracy (r2_score) Using Ridge: {r2_score(y_test, y_test_pred_Ridge)}')
            logger.info(f'Test Loss (Mean_squared_error) Using Ridge: {mean_squared_error(y_test, y_test_pred_Ridge)}')

            #Lasso Regression
            logger.info('Lasso Regression')
            reg_lasso = Lasso(alpha=0.5)
            reg_lasso.fit(X_train, y_train)
            logger.info(f'Intercept : {reg_lasso.intercept_}')
            logger.info(f'Coefficient : {reg_lasso.coef_}')
            y_train_pred_Lasso = reg_lasso.predict(X_train)
            y_test_pred_Lasso = reg_lasso.predict(X_test)
            c ['y_train_pred_Lasso'] =  y_train_pred_Lasso
            logger.info(f'{c.sample(10)}')
            logger.info(f'Training Accuracy (r2_score) Using Lasso: {r2_score(y_train, y_train_pred_Lasso)}')
            logger.info(f'Training Loss (Mean_squared_error) Using Lasso: {mean_squared_error(y_train, y_train_pred_Lasso)}')
            logger.info(f'Test Accuracy (r2_score) Using Lasso: {r2_score(y_test, y_test_pred_Lasso)}')
            logger.info(f'Test Loss (Mean_squared_error) Using Lasso: {mean_squared_error(y_test, y_test_pred_Lasso)}')
            '''
            with open("calories.pkl", 'wb') as f:
                pickle.dump(reg, f)

            with open("calories.pkl", 'rb') as f1:
                m = pickle.load(f1)

            with open("scaler.pkl", 'rb') as f2:
                s = pickle.load(f2)

            t = s.transform([[20,1,190,90,29,105,40]])
            logger.info(f'Prediction {m.predict(t)}')
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')