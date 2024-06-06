import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score as acc_score, confusion_matrix
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show

def read_data(year, fold):
    """
    Read the processed data files for a specific year and fold.

    Parameters:
    - year (int): The year of the data.
    - fold (int): The fold of the data.

    Returns:
    - X_train (DataFrame): The training data.
    - X_test (DataFrame): The testing data.
    - y_train (DataFrame): The training labels.
    - y_test (DataFrame): The testing labels.
    """
    X_train = pd.read_csv(f'processed_data/X_train_{year}_{fold}.csv')
    X_test = pd.read_csv(f'processed_data/X_test_{year}_{fold}.csv')
    y_train = pd.read_csv(f'processed_data/y_train_{year}_{fold}.csv')
    y_test = pd.read_csv(f'processed_data/y_test_{year}_{fold}.csv')

    return X_train, X_test, y_train, y_test

def calculate_metrics(y_test, y_pred_proba, y_pred):
    """
    Calculate various evaluation metrics based on the predicted and actual values.

    Parameters:
    - y_test (array-like): The actual target values.
    - y_pred_proba (array-like): The predicted probabilities for the positive class.
    - y_pred (array-like): The predicted target values.

    Returns:
    - accuracy (float): The accuracy of the predictions.
    - specificity (float): The specificity of the predictions.
    - sensitivity (float): The sensitivity of the predictions.
    - f1 (float): The F1 score of the predictions.
    - auc (float): The area under the ROC curve of the predictions.
    """
    accuracy = acc_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return accuracy, specificity, sensitivity, f1, auc

def update_scores(metrics, accuracy_scores, specificity_scores, sensitivity_scores, f1_scores, auc_scores):
    """
    Updates the scores lists with the given metrics.

    Args:
        metrics (list): A list of metrics to be added to the scores lists.
        accuracy_scores (list): The list of accuracy scores.
        specificity_scores (list): The list of specificity scores.
        sensitivity_scores (list): The list of sensitivity scores.
        f1_scores (list): The list of F1 scores.
        auc_scores (list): The list of AUC scores.

    Returns:
        tuple: A tuple containing the updated accuracy_scores, specificity_scores,
               sensitivity_scores, f1_scores, and auc_scores lists.
    """
    accuracy_scores.append(metrics[0])
    specificity_scores.append(metrics[1])
    sensitivity_scores.append(metrics[2])
    f1_scores.append(metrics[3])
    auc_scores.append(metrics[4])
    return accuracy_scores, specificity_scores, sensitivity_scores, f1_scores, auc_scores
    
       
def create_boxplots(df, metrics_name):
    """
    Create boxplots to visualize the distribution of a metric across different years.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame containing the metric values.
    - metrics_name (str): The name of the metric to be visualized.

    Returns:
    None
    """
    df_melted = pd.melt(df, var_name='Años', value_name=metrics_name)
    metrics_summary = df_melted.groupby('Años')[metrics_name].agg(['mean', 'std']).reset_index()
    print(metrics_name, 'summary across folds:') 
    print(metrics_summary)

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Años', y=metrics_name, data=df_melted)
    plt.title(f'Boxplot of {metrics_name} Across Folds')
    plt.show()

def plot_boxplots(metric_scores, metric_name):
    """
    Plots boxplots for each metric score.

    Args:
        metric_scores (list): A list of lists containing metric scores for each metric.
        metric_name (list): A list of metric names.

    Returns:
        None
    """
    for metric in range(len(metric_scores)):
        df = pd.DataFrame({f'Año {i}': scores for i, scores in enumerate(metric_scores[metric], start=1)})
        create_boxplots(df, metric_name[metric])
        plt.show()

def train(years, n_folds, random_state=1234):   
    """
    Trains a model using Explainable Boosting Classifier (EBM) and calculates various evaluation metrics.

    Parameters:
    - years (int): The number of years to train the model for.
    - n_folds (int): The number of folds for cross-validation.
    - random_state (int): The random seed for reproducibility. Default is 1234.

    Returns:
    - all_accuracy_scores (list): List of accuracy scores for each fold and year.
    - all_specificity_scores (list): List of specificity scores for each fold and year.
    - all_sensitivity_scores (list): List of sensitivity scores for each fold and year.
    - all_f1_scores (list): List of F1 scores for each fold and year.
    - all_auc_scores (list): List of AUC scores for each fold and year.
    """
    all_accuracy_scores = []
    all_specificity_scores = []
    all_sensitivity_scores = []
    all_f1_scores = []
    all_auc_scores = []
    for i in range(1,years+1):
        auc_scores = []
        accuracy_scores = []
        specificity_scores = []
        sensitivity_scores = []
        f1_scores = []
        ebm = ExplainableBoostingClassifier(interactions=0, random_state=random_state)
        for j in range(1, n_folds+1):
            X_train, X_test, y_train, y_test = read_data(i, j)
        
            ebm.fit(X_train, y_train)
            
            y_pred_proba = ebm.predict_proba(X_test)[:, 1]
            y_pred = ebm.predict(X_test)
            
            y_pred = y_pred.astype(float)
            metrics = calculate_metrics(y_test, y_pred_proba, y_pred)
            metrics = update_scores(
                metrics, accuracy_scores, specificity_scores, sensitivity_scores, f1_scores, auc_scores)
            
        metrics = update_scores(
            metrics, all_accuracy_scores, all_specificity_scores, all_sensitivity_scores, all_f1_scores, all_auc_scores)    

    return all_accuracy_scores, all_specificity_scores, all_sensitivity_scores, all_f1_scores, all_auc_scores
     


def main():
    metrics_names = ["Accuracy", "Specificity", "Sensitivity", "F1 Score", "AUC"]
    all_metrics_scores = train(5,5)
    plot_boxplots(all_metrics_scores, metrics_names)

if __name__ == "__main__":
    main()
