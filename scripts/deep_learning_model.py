import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# LSTM Model Class
class LSTMModelBuilder:
    def __init__(self, df, n_lag=14):
        self.df = df
        self.n_lag = n_lag
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
    
    def check_stationarity(self):
        """
        Checks if the 'Sales' data is stationary using the ADF test. If non-stationary (p > 0.05), applies 
        differencing and saves the result in 'Sales_diff'. If stationary, no differencing is applied.
        """
        result = adfuller(self.df['Sales'])
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        if result[1] > 0.05:
            print("The data is non-stationary, differencing is required.")
            self.df['Sales_diff'] = self.df['Sales'].diff().dropna()
            self.df = self.df.dropna()
        else:
            print("The data is stationary.")
            self.df['Sales_diff'] = self.df['Sales']  # No need for differencing