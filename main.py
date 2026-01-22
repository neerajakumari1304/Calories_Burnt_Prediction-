'''
In this file we are going to load the data and other ML pipeline techniques
which are needed
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import os
import seaborn as sns
import warnings

import balance_data

warnings.filterwarnings('ignore')
import pickle

from log_code import setup_logging
logger = setup_logging('main')

from sklearn.model_selection import train_test_split
from missing_value import random_sample
from variable_transform import VT_OUT
from feature_selection import FeatureSelection
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from balance_data import BALANCING_DATA
from model import LR_MODEL

class CALORIES_BURNT_PREDICT:
    def __init__(self,calories_file, exercise_file):
        try:
            self.calories_df = pd.read_csv(calories_file)
            self.exercise_df = pd.read_csv(exercise_file)

            logger.info('Data loaded succesfully')
            self.df = pd.merge(self.exercise_df, self.calories_df, on='User_ID', how='inner')
            self.df = self.df.drop(columns=['User_ID'])
            logger.info(f'After merging{self.df.sample(10)}')
            logger.info(f'{self.df.shape}')
            self.df.reset_index(drop=True, inplace=True)
            logger.info(f'{self.df.isnull().sum()}')

            # Independent and Dependent variables
            self.X = self.df.iloc[:, :-1]
            self.y = self.df.iloc[:, -1]

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                    random_state=42)

            logger.info(f'{self.y_train.sample(5)}')
            logger.info(f'{self.y_test.sample(5)}')

            logger.info(f'Training data size : {self.X_train.shape}')
            logger.info(f'Testing data size : {self.X_test.shape}')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

    def missing_values(self):
        try:
            missing_value = setup_logging('missing_value')
            missing_value.info(f'total rows in training data: {self.X_train.shape}')
            missing_value.info(f'total rows in testing data: {self.X_test.shape}')
            missing_value.info(f"Before : {self.X_train.columns}")
            missing_value.info(f"Before : {self.X_test.columns}")
            missing_value.info(f"Before : {self.X_train.isnull().sum()}")
            missing_value.info(f"Before : {self.X_test.isnull().sum()}")
            self.X_train, self.X_test = random_sample.random_sample_imputation_technique(self.X_train, self.X_test)
            missing_value.info(f"After : {self.X_train.columns}")
            missing_value.info(f"After : {self.X_test.columns}")
            missing_value.info(f"After : {self.X_train.isnull().sum()}")
            missing_value.info(f"After : {self.X_test.isnull().sum()}")
            missing_value.info(f'total rows in training data: {self.X_train.shape}')
            missing_value.info(f'total rows in testing data: {self.X_test.shape}')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

    def vt_out(self):
        try:
            vt_out = setup_logging('variable_transform')
            vt_out.info('Variable Transformation and Outlier Detection')

            for i in self.X_train.columns:
                vt_out.info(f'{self.X_train[i].dtype}')

            vt_out.info(f'{self.X_train.columns}')
            vt_out.info(f'{self.X_test.columns}')

            # Call transformation
            self.X_train, self.X_test = VT_OUT.variable_transformation_outliers(self.X_train, self.X_test)

            vt_out.info(f'{self.X_train.columns} --> {self.X_train.shape}')
            vt_out.info(f'{self.X_test.columns} --> {self.X_test.shape}')

            vt_out.info('Variable Transformation Completed')

            # Return the updated attributes
            return self.X_train, self.X_test

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')
            return self.X_train, self.X_test

    def fs(self):
        try:
            fs = setup_logging('feature_selection')
            fs.info(f"Before : {self.X_train.columns}->{self.X_train.shape}")
            fs.info(f"Before : {self.X_test.columns}->{self.X_test.shape}")
            self.X_train, self.X_test = FeatureSelection.run(self.X_train, self.X_test,
                                                                           self.y_train)
            fs.info(f"After : {self.X_train.columns}->{self.X_train.shape}")
            fs.info(f"After : {self.X_test.columns}->{self.X_test.shape}")

            return self.X_train, self.X_test

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')
            return self.X_train, self.X_test

    def cat_to_num(self):
        cat = setup_logging('cat_to_num')
        try:
            cat.info('Categorical to Numerical')
            cat.info(f'{self.X_train.columns}')
            cat.info(f'{self.X_test.columns}')

            for i in self.X_train.columns:
                cat.info(f'{i} --> {self.X_train[i].unique()}')

            cat.info(f'Before Converting : {self.X_train}')
            cat.info(f'Before Converting : {self.X_test}')

            # One-Hot Encoding
            one_hot = OneHotEncoder(drop='first')
            one_hot.fit(self.X_train[['Gender']])

            # Transform train
            train_encoded = one_hot.transform(self.X_train[['Gender']]).toarray()
            train_encoded_df = pd.DataFrame(train_encoded, columns=one_hot.get_feature_names_out())
            self.X_train_cat = pd.concat(
                [self.X_train.drop(['Gender'], axis=1).reset_index(drop=True),
                 train_encoded_df.reset_index(drop=True)], axis=1
            )

            # Transform test
            test_encoded = one_hot.transform(self.X_test[['Gender']]).toarray()
            test_encoded_df = pd.DataFrame(test_encoded, columns=one_hot.get_feature_names_out())
            self.X_test_cat = pd.concat(
                [self.X_test.drop(['Gender'], axis=1).reset_index(drop=True),
                 test_encoded_df.reset_index(drop=True)], axis=1
            )

            self.X_train = self.X_train_cat
            self.X_test = self.X_test_cat

            cat.info(f'{self.X_train_cat.columns}')
            cat.info(f'{self.X_test_cat.columns}')

            cat.info(f"After Converting : {self.X_train_cat}")
            cat.info(f"After Converting : {self.X_test_cat}")

            cat.info(f"{self.X_train_cat.shape}")
            cat.info(f"{self.X_test_cat.shape}")

            cat.info(f"{self.X_train_cat.isnull().sum()}")
            cat.info(f"{self.X_test_cat.isnull().sum()}")

            self.X_train.reset_index(drop=True, inplace=True)
            self.X_train_cat.reset_index(drop=True, inplace=True)

            self.X_test.reset_index(drop=True, inplace=True)
            self.X_test_cat.reset_index(drop=True, inplace=True)

            self.training_data = pd.concat([self.X_train, self.X_train_cat], axis=1)
            self.testing_data = pd.concat([self.X_test, self.X_test_cat], axis=1)

            cat.info(f"{self.training_data.shape}")
            cat.info(f"{self.testing_data.shape}")

            cat.info(f"{self.training_data.isnull().sum()}")
            cat.info(f"{self.testing_data.isnull().sum()}")

            cat.info(f"=======================================================")

            cat.info(f"Training Data : {self.training_data.sample(10)}")
            cat.info(f"Testing Data : {self.testing_data.sample(10)}")

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

    def balance(self):
        try:
            bal = setup_logging('balance_data')
            bal.info('Scaling Data Before Regression')
            self.X_train, self.X_test = BALANCING_DATA.balance(self.X_train, self.X_test)

            LR_MODEL.linear_regression(self.X_train, self.y_train, self.X_test, self.y_test)
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')


if __name__ == "__main__":
    try:

        obj = CALORIES_BURNT_PREDICT('D:\\Calories_Burnt\\pythonProject1\\calories (1).csv','D:\\Calories_Burnt\\pythonProject1\\exercise.csv')
        obj.missing_values()
        obj.vt_out()
        obj.fs()
        obj.cat_to_num()
        obj.balance()

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')