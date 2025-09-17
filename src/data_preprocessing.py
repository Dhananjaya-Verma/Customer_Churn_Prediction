# src/data_preprocessing.py
import pandas as pd
import numpy as np

def load_data(path):
    df = pd.read_csv(r"D:\CSE(DataScience)\Customer_Churn_Prediction\Datasets\Telco-Customer-Churn-dataset.csv")
    return df

def clean_basic(df):
    # Convert TotalCharges and fill NA
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    return df

def split_xy(df, target='Churn'):
    df[target] = df[target].map({'Yes':1, 'No':0}) if df[target].dtype == 'object' else df[target]
    X = df.drop(target, axis=1)
    y = df[target]
    return X, y