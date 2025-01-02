import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from scipy import stats
import holidays
from dateutil import easter
import logging

class CustomerBehaviorAnalyzer:
    def __init__(self, data):
        self.data = data
    # Setup logging configuration
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
    def merge_data(self, train, test, store):
        """
        Merges the train and test data with the store data on the store column
        Args:
            train (pandas.DataFrame): Train data
            test (pandas.DataFrame): Test data
            store (pandas.DataFrame): Store data
            
        Returns:
            tuple: (merged_train, merged_test)
        """
        logging.info("Merging train and test data with store data")
        merged_train = pd.merge(train, store, how='left', on='Store')
        merged_test = pd.merge(test, store,how='left', on='Store')
        return merged_train, merged_test
    # Data cleaning function
    def clean_data(self, df):
        """
        Cleans the input DataFrame by handling missing values and removing outliers.
        Args:
            df (pandas.DataFrame): Input DataFrame
        Returns:
            pandas.DataFrame: Cleaned DataFrame
        """
        logging.info("Cleaning data")
        # Handling missing values
        imputer = SimpleImputer(strategy='mean')
        df['CompetitionDistance'] = imputer.fit_transform(df[['CompetitionDistance']])
        
        # Outlier detection using Z-score for 'Sales' and 'Customers'
        df = df[(np.abs(stats.zscore(df[['Sales', 'Customers']])) < 3).all(axis=1)]
        
        return df
    def handle_catagorical_values(self, df,columns):
        """
        Handles categorical values in the dataset by converting them to numerical using one-hot encoding.
        Args:
            df (pandas.DataFrame): Input DataFrame
            columns (list): List of column names to be encoded
        Returns:
            pandas.DataFrame: DataFrame with categorical values encoded
        """
        logging.info("Handling categorical values")
        # Convert categorical variables to numerical using one-hot encoding
        df = pd.get_dummies(df, columns)
        return df
    
    # Plotting the distribution of promotions in training and test sets
    def plot_promo_distribution(self, merge_train_df, merge_test_df):
        """
        Plots the distribution of promotions in the training and test sets.
        Args:
            merge_train_df (pandas.DataFrame): Merged train data
            merge_test_df (pandas.DataFrame): Merged test data
        """
        logging.info("Plotting promotion distribution")
        # Plotting the distribution of promotions
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        sns.histplot(merge_train_df['Promo'], kde=False, ax=ax[0])
        ax[0].set_title('Promo Distribution - Training Set')
        
        sns.histplot(merge_test_df['Promo'], kde=False, ax=ax[1], color='green')
        ax[1].set_title('Promo Distribution - Test Set')
        
        plt.show()
    