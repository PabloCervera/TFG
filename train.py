import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score as acc_score, confusion_matrix
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show

def calculate_metrics(y_test, y_pred_proba, y_pred):
    # Calcula las métricas de rendimiento
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
    # Añade las métricas de rendimiento a las listas
    accuracy_scores.append(metrics[0])
    specificity_scores.append(metrics[1])
    sensitivity_scores.append(metrics[2])
    f1_scores.append(metrics[3])
    auc_scores.append(metrics[4])
    return accuracy_scores, specificity_scores, sensitivity_scores, f1_scores, auc_scores
    
def read_data(year, fold):
    # Lee el archivo CSV
    X_train = pd.read_csv(f'processed_data/X_train_{year}_{fold}.csv')
    X_test = pd.read_csv(f'processed_data/X_test_{year}_{fold}.csv')
    y_train = pd.read_csv(f'processed_data/y_train_{year}_{fold}.csv')
    y_test = pd.read_csv(f'processed_data/y_test_{year}_{fold}.csv')

    return X_train, X_test, y_train, y_test    
       
def create_boxplots(df, metrics_name):
    df_melted = pd.melt(df, var_name='Años', value_name=metrics_name)
    df_melted.to_csv(f'{metrics_name}.csv', index=False)
    metrics_summary = df_melted.groupby('Años')[metrics_name].agg(['mean', 'std']).reset_index()
    print(metrics_name, 'summary across folds:') 
    print(metrics_summary)

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Años', y=metrics_name, data=df_melted)
    plt.title(f'Boxplot of {metrics_name} Across Folds')
    plt.show()

def plot_boxplots(metric_scores, metric_name):
    for metric in range(len(metric_scores)):
        df = pd.DataFrame({f'Año {i}': scores for i, scores in enumerate(metric_scores[metric], start=1)})
        create_boxplots(df, metric_name[metric])
        plt.show()

def train(years, n_folds, random_state=1234):   
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
