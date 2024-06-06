import sys
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd

import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
from missingpy import MissForest
    
def imputed_data(X_train, X_test, random_state=1234):
    """
    Imputes missing values in the input data using the MissForest algorithm.

    Parameters:
    X_train (array-like): The training data with missing values.
    X_test (array-like): The test data with missing values.
    random_state (int, optional): Random seed for reproducibility. Defaults to 1234.

    Returns:
    tuple: A tuple containing the imputed training data and imputed test data.
    """

    imputer = MissForest(random_state=random_state)
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.fit_transform(X_test)
    
    return X_train_imputed, X_test_imputed

def balance_classes(X_train, y_train, random_state=1234):
    """
    Balances the classes in the training data using the SMOTE algorithm.

    Parameters:
    - X_train (array-like): The input features of the training data.
    - y_train (array-like): The target labels of the training data.
    - random_state (int): The random seed for reproducibility. Default is 1234.

    Returns:
    - X_train_resampled (array-like): The resampled input features with balanced classes.
    - y_train_resampled (array-like): The resampled target labels with balanced classes.
    """
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    return X_train_resampled, y_train_resampled

def convert_to_csv(X, X_train, X_test, y_train, y_test, year, fold):
    """
    Converts the input data and labels into CSV files and saves them in the 'processed_data' directory.

    Args:
        X (pd.DataFrame): The input data.
        X_train (np.ndarray): The training data.
        X_test (np.ndarray): The testing data.
        y_train (np.ndarray): The training labels.
        y_test (np.ndarray): The testing labels.
        year (int): The year of the data.
        fold (int): The fold number.

    Returns:
        None
    """
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    y_train = pd.DataFrame(y_train, columns=['pred'])
    y_test = pd.DataFrame(y_test, columns=['pred'])
    if not os.path.exists('processed_data'):
        os.mkdir('processed_data')
    X_train.to_csv(f'processed_data/X_train_{year}_{fold}.csv', index=False)
    X_test.to_csv(f'processed_data/X_test_{year}_{fold}.csv', index=False)
    y_train.to_csv(f'processed_data/y_train_{year}_{fold}.csv', index=False)
    y_test.to_csv(f'processed_data/y_test_{year}_{fold}.csv', index=False)

def preprocess_data(X, y, n_folds, year, random_state=1234):
    """
    Preprocesses the data by performing imputation, class balancing, and conversion to CSV.

    Args:
        X (pandas.DataFrame): The input features.
        y (numpy.ndarray): The target variable.
        n_folds (int): The number of folds for cross-validation.
        year (int): The year for which the data is being preprocessed.
        random_state (int, optional): The random seed for reproducibility. Defaults to 1234.

    Returns:
        None
    """
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)        
    
    for j, (train_i, test_i) in enumerate(kf.split(X), start=1):
        X_train, X_test = X.iloc[train_i, :], X.iloc[test_i,:]
        y_train, y_test = y[train_i], y[test_i]   
    
        X_train, X_test = imputed_data(X_train, X_test)
        
        X_train_resampled, y_train_resampled = balance_classes(X_train, y_train)

        convert_to_csv(X, X_train_resampled, X_test, y_train_resampled, y_test, year, j)   
            
def main():   
    filename = f'data/X.csv'
    df = pd.read_csv(filename)
    for i in range(1, 6):     
        y = pd.read_csv(f'data/y_{i}.csv')

        X = pd.merge(df, y, on='RID', how='inner')
        
        y = X['DX']
        X = X.drop(columns=['RID','VISCODE','DX'])
        
        y = y.replace( 'Dementia', 1 )
        y = y.replace( 'MCI', 0 )
        
        y = y.values
        
        preprocess_data(X, y, 5, i)
    


if __name__ == "__main__":
    main()