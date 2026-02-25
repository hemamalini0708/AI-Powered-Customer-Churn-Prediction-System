import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import os
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from log_code import phase_1
logger = phase_1('imbalance_data')
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import pickle
from all_models import common

def balance_data(X_train,y_train,X_test,y_test):
    try:
        logger.info(f'Before Number of rows for Good class : {sum(y_train == 1)}')
        logger.info(f'Before Number of rows for Bad class : {sum(y_train == 0)}')
        sm_res = SMOTE(random_state=42)
        X_train_bal,y_train_bal = sm_res.fit_resample(X_train,y_train)
        logger.info(f'After Number of rows for Good class : {sum(y_train_bal == 1)}')
        logger.info(f'After Number of rows for Bad class : {sum(y_train_bal == 0)}')

        logger.info(f"{X_train_bal.shape}")
        logger.info(f'{y_train_bal.shape}')
        logger.info(f'{X_train_bal.sample(10)}')
        sc = StandardScaler()
        sc.fit(X_train_bal)
        X_train_bal_scaled = sc.transform(X_train_bal)
        X_test_scaled = sc.transform(X_test)
        logger.info(f'{X_train_bal_scaled}')
        with open('scalar.pkl','wb') as f:
            pickle.dump(sc,f)

        common(X_train_bal_scaled,y_train_bal,X_test_scaled,y_test)

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in line no : {error_line.tb_lineno} : due to {error_msg}')
