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
            self.df['Sales_diff'] = self.df['Sales']
    
    def plot_acf_pacf(self):
        """
        Plots the ACF and PACF for the differenced 'Sales' data ('Sales_diff') with 50 lags.
        """
        plot_acf(self.df['Sales_diff'], lags=50)
        plot_pacf(self.df['Sales_diff'], lags=50)
        plt.show()

    def create_supervised_data(self):
        """
        Transforms the differenced 'Sales' data into supervised learning format 
        by creating input-output pairs based on the specified lag.
        """
        X, y = [], []
        data = self.df['Sales_diff'].values
        for i in range(len(data) - self.n_lag):
            X.append(data[i:i + self.n_lag])
            y.append(data[i + self.n_lag])
        return np.array(X), np.array(y)
    
    def scale_data(self, X, y):
        """
        Scales features (X) and target (y) using MinMaxScaler.

        Parameters:
            X (ndarray): Input features to be scaled.
            y (ndarray): Target variable to be scaled.

        Returns:
            tuple: Scaled features (X_scaled) and scaled target (y_scaled).
        """
        X_scaled = self.scaler.fit_transform(X)
        y_scaled = self.scaler.fit_transform(y.reshape(-1, 1))
        return X_scaled, y_scaled
    def split_data(self, X, y, train_size=0.8):
        """
        Split the data into train and test sets.

        Parameters:
            X (ndarray): Input features.
            y (ndarray): Target variable.
            train_size (float): Proportion of data to be used for training (default is 0.8).
        Returns:
            tuple: X_train, X_test, y_train, y_test split into training and testing sets.
        """
        train_size = int(len(X) * train_size)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Reshape the data to be 3D for LSTM (samples, time steps, features)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        return X_train, X_test, y_train, y_test
