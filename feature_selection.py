import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from scipy.stats import pearsonr
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

from log_code import setup_logging
logger = setup_logging('feature_selection')


class FeatureSelection:
    @staticmethod
    def run(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series):
        try:
            if X_train.shape[1] == 0:
                logger.info("No features available for selection.")
                return X_train, X_test

            # Step 1: Remove constant features
            reg_con = VarianceThreshold(threshold=0.0)
            X_train = pd.DataFrame(reg_con.fit_transform(X_train),
                                   columns=X_train.columns[reg_con.get_support()])
            X_test = pd.DataFrame(reg_con.transform(X_test),
                                  columns=X_test.columns[reg_con.get_support()])
            logger.info(f"After removing constant features: Train {X_train.shape}, Test {X_test.shape}")

            # Step 2: Remove quasi-constant features
            reg_quasi = VarianceThreshold(threshold=0.01)
            X_train = pd.DataFrame(reg_quasi.fit_transform(X_train),
                                   columns=X_train.columns[reg_quasi.get_support()])
            X_test = pd.DataFrame(reg_quasi.transform(X_test),
                                  columns=X_test.columns[reg_quasi.get_support()])
            logger.info(f"After removing quasi-constant features: Train {X_train.shape}, Test {X_test.shape}")

            # Step 3: Pearson correlation with target (Calories)
            corr_df = pd.DataFrame({
                "correlation": [pearsonr(X_train[col], y_train)[0] for col in X_train.columns],
                "p_value": [pearsonr(X_train[col], y_train)[1] for col in X_train.columns]
            }, index=X_train.columns)

            logger.info(f"Correlation summary:\n{corr_df}")

            # Step 4: Drop weak features (p-value > 0.05)
            weak_features = corr_df[corr_df["p_value"] > 0.05].index
            X_train = X_train.drop(columns=weak_features)
            X_test = X_test.drop(columns=weak_features)
            logger.info(f"Removed weak features: {list(weak_features)}")
            logger.info(f"Final shapes: Train {X_train.shape}, Test {X_test.shape}")

            return X_train, X_test

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')
            return X_train, X_test

