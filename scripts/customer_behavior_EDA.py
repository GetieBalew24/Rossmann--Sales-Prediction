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
    