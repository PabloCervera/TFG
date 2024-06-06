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
    imputer = MissForest(random_state=random_state)
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.fit_transform(X_test)
    
    return X_train_imputed, X_test_imputed

def balance_classes(X_train, y_train, random_state=1234):
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    return X_train_resampled, y_train_resampled

def convert_to_csv(X, X_train, X_test, y_train, y_test, year, fold):
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
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)        
    
    for j, (train_i, test_i) in enumerate(kf.split(X), start=1):
        X_train, X_test = X.iloc[train_i, :], X.iloc[test_i,:]
        y_train, y_test = y[train_i], y[test_i]   
    
        # Imputa los valores faltantes con MissForest
        X_train, X_test = imputed_data(X_train, X_test)
        
        # Aplica SMOTE para balanceo de clases          
        X_train_resampled, y_train_resampled = balance_classes(X_train, y_train)

        convert_to_csv(X, X_train_resampled, X_test, y_train_resampled, y_test, year, j)
         
            
def main():   
    filename = f'data/X_set.csv'
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