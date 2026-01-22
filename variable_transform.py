import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

import warnings
warnings.filterwarnings('ignore')

from log_code import setup_logging
logger = setup_logging('variable_transform')

class VT_OUT:
    @staticmethod
    def variable_transformation_outliers(X_train: pd.DataFrame, X_test: pd.DataFrame):
        try:
            # ✅ Log initial info
            logger.info(f"Train columns: {X_train.columns} -> {X_train.shape}")
            logger.info(f"Test columns: {X_test.columns} -> {X_test.shape}")
            logger.info("Before Variable Transformation")

            # ✅ Ensure plot folder exists
            PLOT_PATH = "plots_path"
            os.makedirs(PLOT_PATH, exist_ok=True)

            # ---------- BEFORE TRANSFORMATION ----------
            for col in X_train.columns:
                plt.figure()
                X_train[col].plot(kind='kde', color='r')
                plt.title(f'KDE-{col} (Before)')
                plt.savefig(f'{PLOT_PATH}/kde_before_{col}.png')
                plt.close()

                plt.figure()
                sns.boxplot(x=X_train[col])
                plt.title(f'Boxplot-{col} (Before)')
                plt.savefig(f'{PLOT_PATH}/boxplot_before_{col}.png')
                plt.close()

                # ---------- LOG TRANSFORM (all numeric columns) ----------
            for col in X_train.select_dtypes(include=['int64', 'float64']).columns:
                X_train[col] = np.log1p(X_train[col])
                X_test[col] = np.log1p(X_test[col])

                # ---------- AFTER TRANSFORMATION ----------
            for col in X_train.columns:
                plt.figure()
                sns.boxplot(x=X_train[col])
                plt.title(f'Boxplot-{col} (After)')
                plt.savefig(f'{PLOT_PATH}/boxplot_after_{col}.png')
                plt.close()

            logger.info(f"Train columns: {X_train.columns} -> {X_train.shape}")
            logger.info(f"Test columns: {X_test.columns} -> {X_test.shape}")
            logger.info("After Variable Transformation")

            return X_train, X_test
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')
            return X_train, X_test
