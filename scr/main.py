'''
Customer Churn Prediction - Main Training Script
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import os
import seaborn as sns
import warnings
import pickle                     # <-- ADDED (for saving objects)
warnings.filterwarnings('ignore')
from log_code import phase_1
logger = phase_1('main')
from sklearn.model_selection import train_test_split
from random_sample_imputation import random_sample_imputation_technique
from var_trnsf_outlrs import variable_transformation_outliers
from feature_selection import complete_feature_selection
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from imbalance_data import balance_data

class CUSTOMER_CHURN_DATA:
    def __init__(self, path):
        try:
            self.path = path
            self.df = pd.read_csv(self.path)
            logger.info(f'Data Loaded Sucessfully')
            logger.info(f'Total Rows : {self.df.shape[0]}')
            logger.info(f'Total Columns : {self.df.shape[1]}')
            logger.info(f'{self.df.isnull().sum()}')
            logger.info(f'Before')
            logger.info(f'{self.df.dtypes}')
            logger.info(f'After')
            self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
            self.df = self.df.drop(columns=['customerID'])
            binary_cols = ['Churn', 'PhoneService']
            self.df[binary_cols] = self.df[binary_cols].replace({'Yes': 1, 'No': 0})
            logger.info(f'{self.df.dtypes}')
            logger.info(f'{self.df.isnull().sum()}')
            self.X = self.df.iloc[:, :-1]
            self.y = self.df.iloc[:, -1]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                            test_size=0.2,
                                                                                            random_state=42)
            logger.info(f'Training data size : {self.X_train.shape}')
            logger.info(f'Testing data size : {self.X_test.shape}')

            logger.info(f'{self.df.columns}')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no : {error_line.tb_lineno} : due to {error_msg}')

    def missing_values(self):
        try:
            self.X_train, self.X_test = random_sample_imputation_technique(self.X_train, self.X_test)
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no : {error_line.tb_lineno} : due to {error_msg}')

    def var_trns_out(self):
        try:
            logger.info(f'{self.X_train.columns}')
            logger.info(f'{self.X_test.columns}')
            self.X_train_num = self.X_train.select_dtypes(exclude='object')
            self.X_train_cat = self.X_train.select_dtypes(include='object')
            self.X_test_num = self.X_test.select_dtypes(exclude='object')
            self.X_test_cat = self.X_test.select_dtypes(include='object')
            logger.info(f'{self.X_train_num.columns}')
            logger.info(f'{self.X_train_cat.columns}')
            logger.info(f'{self.X_test_num.columns}')
            logger.info(f'{self.X_test_cat.columns}')

            logger.info(f'{self.X_train_num.shape}')
            logger.info(f'{self.X_train_cat.shape}')
            logger.info(f'{self.X_test_num.shape}')
            logger.info(f'{self.X_test_cat.shape}')
            self.X_train_num, self.X_test_num = variable_transformation_outliers(self.X_train_num,
                                                                                         self.X_test_num)
            logger.info(f"{self.X_train_num.columns} -> {self.X_train_num.shape}")
            logger.info(f"{self.X_test_num.columns} -> {self.X_test_num.shape}")
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no : {error_line.tb_lineno} : due to {error_msg}')

    def feature_s(self):
        try:
            logger.info(f"Before : {self.X_train_num.columns} -> {self.X_train_num.shape}")
            logger.info(f"Before : {self.X_test_num.columns} -> {self.X_test_num.shape}")
            self.X_train_num, self.X_test_num = complete_feature_selection(self.X_train_num, self.X_test_num,
                                                                                   self.y_train)
            logger.info(f"After : {self.X_train_num.columns} -> {self.X_train_num.shape}")
            logger.info(f"After : {self.X_test_num.columns} -> {self.X_test_num.shape}")

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no : {error_line.tb_lineno} : due to {error_msg}')

    def cat_num(self):
        try:
            logger.info(f'{self.X_train_cat.columns}')
            logger.info(f'{self.X_test_cat.columns}')
            for i in self.X_train_cat.columns:
                logger.info(f"{i} -> : {self.X_train_cat[i].unique()}")

            logger.info(f'Before Converting : {self.X_train_cat.columns}')
            logger.info(f'Before Converting : {self.X_test_cat.columns}')
            logger.info(f'Before Converting : {self.X_train_cat}')
            logger.info(f'Before Converting : {self.X_test_cat}')

            one_hot = OneHotEncoder(drop='first')
            one_hot.fit(self.X_train_cat[["gender", "Partner"]])
            # <-- ADDED: save one-hot encoder
            with open('one_hot.pkl', 'wb') as f:
                pickle.dump(one_hot, f)

            result = one_hot.transform(self.X_train_cat[["gender", "Partner"]]).toarray()
            f = pd.DataFrame(data=result, columns=one_hot.get_feature_names_out())
            self.X_train_cat.reset_index(drop=True, inplace=True)
            f.reset_index(drop=True, inplace=True)
            self.X_train_cat = pd.concat([self.X_train_cat, f], axis=1)
            self.X_train_cat = self.X_train_cat.drop(['gender', 'Partner'], axis=1)

            result1 = one_hot.transform(self.X_test_cat[["gender", "Partner"]]).toarray()
            f1 = pd.DataFrame(data=result1, columns=one_hot.get_feature_names_out())
            self.X_test_cat.reset_index(drop=True, inplace=True)
            f1.reset_index(drop=True, inplace=True)
            self.X_test_cat = pd.concat([self.X_test_cat, f1], axis=1)
            self.X_test_cat = self.X_test_cat.drop(["gender", "Partner"], axis=1)

            ord_cols = [
                "Dependents", "MultipleLines", "InternetService", "OnlineSecurity",
                "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
                "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "sim"
            ]
            ord_enc = OrdinalEncoder()
            ord_enc.fit(self.X_train_cat[ord_cols])
            # <-- ADDED: save ordinal encoder
            with open('ord_enc.pkl', 'wb') as f:
                pickle.dump(ord_enc, f)

            result2 = ord_enc.transform(self.X_train_cat[ord_cols])
            t = pd.DataFrame(
                   data=result2,
                   columns=[col + "_res" for col in ord_cols]
            )
            self.X_train_cat.reset_index(drop=True, inplace=True)
            t.reset_index(drop=True, inplace=True)
            self.X_train_cat = pd.concat([self.X_train_cat, t], axis=1)
            self.X_train_cat.drop(ord_cols, axis=1, inplace=True)

            ord_cols = [
                "Dependents", "MultipleLines", "InternetService",
                "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies",
                "Contract", "PaperlessBilling", "PaymentMethod", "sim"
            ]
            result3 = ord_enc.transform(self.X_test_cat[ord_cols])
            t1 = pd.DataFrame(
                  data=result3,
                  columns=[col + "_res" for col in ord_cols]
            )
            self.X_test_cat.reset_index(drop=True, inplace=True)
            t1.reset_index(drop=True, inplace=True)
            self.X_test_cat = pd.concat([self.X_test_cat, t1], axis=1)
            self.X_test_cat.drop(ord_cols, axis=1, inplace=True)

            logger.info(f'{self.X_train_cat.columns}')
            logger.info(f'{self.X_test_cat.columns}')
            logger.info(f'After Converting : {self.X_train_cat.columns}')
            logger.info(f'After Converting : {self.X_test_cat.columns}')
            logger.info(f'After Converting : {self.X_train_cat}')
            logger.info(f'After Converting : {self.X_test_cat}')

            self.X_train_num.reset_index(drop=True, inplace=True)
            self.X_train_cat.reset_index(drop=True, inplace=True)
            self.X_test_num.reset_index(drop=True, inplace=True)
            self.X_test_cat.reset_index(drop=True, inplace=True)

            self.training_data = pd.concat([self.X_train_num, self.X_train_cat], axis=1)
            self.testing_data = pd.concat([self.X_test_num, self.X_test_cat], axis=1)

            logger.info(f'{self.training_data.shape}')
            logger.info(f'{self.testing_data.shape}')

            logger.info(f'{self.training_data.isnull().sum()}')
            logger.info(f'{self.testing_data.isnull().sum()}')

            # <-- ADDED: save final feature column order
            with open('feature_columns.pkl', 'wb') as f:
                pickle.dump(list(self.training_data.columns), f)

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no : {error_line.tb_lineno} : due to {error_msg}')

    def data_balance(self):
        try:
            def safe_target_mapping(y):
                if y.dtype == 'object':
                    y = y.map({'Yes': 1, 'No': 0})
                return y.astype('Int64')  # nullable integer type

            self.y_train = safe_target_mapping(self.y_train)
            self.y_test = safe_target_mapping(self.y_test)

            balance_data(self.training_data, self.y_train, self.testing_data, self.y_test)

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no : {error_line.tb_lineno} : due to {error_msg}')


if __name__ == "__main__":
    try:
        obj = CUSTOMER_CHURN_DATA('D:\\Users\\geeth\\PycharmProjects\\Telecom_Customer_Churn_Prediction_System\\data\\Sample_data.csv')
        obj.missing_values()
        obj.var_trns_out()
        obj.feature_s()
        obj.cat_num()
        obj.data_balance()
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')