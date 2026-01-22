import numpy as np
import pandas as pd
import sys

from log_code import setup_logging
logger = setup_logging('balance_data')
from sklearn.preprocessing import StandardScaler
import pickle

class BALANCING_DATA:
    @staticmethod
    def balance(X_train, X_test):
        try:
            logger.info('Balancing data')
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )

            logger.info(f'After scaling train: {X_train_scaled.shape}')
            logger.info(f'After scaling test: {X_test_scaled.shape}')
            logger.info(f'Columns: {list(X_train_scaled.columns)}')

            with open('scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)

            logger.info("Scaling completed successfully")
            return X_train_scaled, X_test_scaled



        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Error in line {error_line.tb_lineno}: {error_msg}")
            return X_train, y_train
