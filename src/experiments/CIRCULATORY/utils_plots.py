from sklearn.metrics import confusion_matrix, roc_auc_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix


####################################################################
# FUNCTIONS TO ANALYZE THE RESULTS (SCORES) AND PROBABILITIES #
####################################################################
def plotProbs(y_test_df, y_pred_df, n_time_steps, title, mask, n_pat):
    """
    Plots the predicted values and the percentage of samples over time.
    Args:
        - y_test_df : DataFrame containing the real values.
        - y_pred_df : DataFrame containing the predicted values.
        - n_time_steps: Number of time steps.
        - title: String with the title of the plot.
        - mask: DataFrame indicating which values are masked.
        - n_pat: Total number of samples.

    """
    masked_data = [y_pred_df.iloc[:, t][~mask.iloc[:, t]] for t in range(n_time_steps)]

    filtered_masked_data = [d for d in masked_data if len(d) > 0]

    # Count the number of samples per time step
    num_samples = [len(d) for d in masked_data]

    # Calculate the percentage of samples with respect to the first time step
    percentage_samples = [100 * num_samples[t] / n_pat if n_pat != 0 else 0 for t in range(n_time_steps)]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.boxplot(filtered_masked_data, patch_artist=True)
    ax1.axhline(y=0.5, color='blue', linestyle='--', linewidth=1, label='0.5 Threshold')

    filtered_time_steps = [t for t in range(n_time_steps) if len(masked_data[t]) > 0]
    ax1.set_xticks(range(1, len(filtered_time_steps) + 1))
    ax1.set_xticklabels([f'{i+1}' for i in filtered_time_steps])

    ax2 = ax1.twinx()
    ax2.plot(range(1, len(percentage_samples) + 1), percentage_samples, color='red', marker='o', linestyle='-', linewidth=2, label='Percentage of Samples')
    ax2.set_ylabel('Percentage of Samples (%)')
    ax2.legend(loc='upper right')
    
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
    
def get_metrics_over_time(n_time_steps, y_test_df, y_pred_df):
    """
    Calculate metrics per time step.
    Args:
        - n_time_steps: Number of time steps.
        - y_test_df : DataFrame containing the real values.
        - y_pred_df : DataFrame containing the predicted values.
    Returns:
        - metrics_df: DataFrame containing the metrics for each time step.
    """
    mask = (y_test_df == 666)

    # Lists to store the metrics
    tn_list = []
    fp_list = []
    fn_list = []
    tp_list = []
    specificity_list = []
    recall_list = []
    roc_auc_list = []
    f1_score_list = []
    accuracy_list = []

    # Calculate the metrics for each time step
    for t in range(n_time_steps):
        # Filter the valid values according to the mask
        valid_indices = ~mask.iloc[:, t]
        y_test_valid = y_test_df.iloc[:, t][valid_indices]
        y_pred_valid = y_pred_df.iloc[:, t][valid_indices]

        # Round the predictions
        y_pred_rounded = np.round(y_pred_valid)

        # Calculate the confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test_valid, y_pred_rounded, labels=[0, 1]).ravel()

        # Calculate specificity, recall, accuracy, and F1 score
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else np.nan
        f1_score = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else np.nan

        # Calculate ROC-AUC
        roc = roc_auc_score(y_test_valid, y_pred_valid) if len(np.unique(y_test_valid)) > 1 else np.nan

        tn_list.append(tn)
        fp_list.append(fp)
        fn_list.append(fn)
        tp_list.append(tp)
        specificity_list.append(specificity)
        recall_list.append(recall)
        roc_auc_list.append(roc)
        f1_score_list.append(f1_score)
        accuracy_list.append(accuracy)

    # Dataframe to store the metrics per time step
    metrics_df = pd.DataFrame({
        'Time Step': range(1, n_time_steps+1),
        'TN': tn_list,
        'FP': fp_list,
        'FN': fn_list,
        'TP': tp_list,
        'Specificity': specificity_list,
        'Recall': recall_list,
        'ROC AUC': roc_auc_list,
        'F1 Score': f1_score_list,
        'Accuracy': accuracy_list
    })

    return metrics_df



def plot_metrics_over_time(metrics_df, T):
    """
    Plot metrics over time.
    Args:
        - metrics_df: DataFrame containing the metrics for each time step.
        - T: Number of time steps.
    """

    plt.figure(figsize=(12, 6))

    plt.plot(metrics_df['Time Step'], metrics_df['Specificity'], label='Specificity', marker='o')
    plt.plot(metrics_df['Time Step'], metrics_df['Recall'], label='Sensitivity', marker='s')
    plt.plot(metrics_df['Time Step'], metrics_df['ROC AUC'], label='ROC AUC', marker='^')
    plt.plot(metrics_df['Time Step'], metrics_df['F1 Score'], label='F1 Score', marker='d')
    plt.plot(metrics_df['Time Step'], metrics_df['Accuracy'], label='Accuracy', marker='x')

    plt.legend()
    plt.xticks(ticks=range(1, T + 1))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(0.85, 1.0)
    plt.show()


    
def plot1_metrics_over_time(all_metrics, T, path, save_img=False):
    """
    Plot metrics over time with shaded areas indicating standard deviation.
    Args:
        - all_metrics: A list with the temporal metrics of each split.
        - T: Number of time steps.
    """
    plt.figure(figsize=(12, 6))

    metrics_concat = pd.concat(all_metrics)
    metrics_mean = metrics_concat.groupby('Time Step').mean()
    metrics_std = metrics_concat.groupby('Time Step').std()

    time_steps = metrics_mean.index

    # Plot mean and standard deviation for each metric
    plt.plot(time_steps, metrics_mean['Specificity'], label='Specificity', marker='o')
    plt.fill_between(time_steps,
                     metrics_mean['Specificity'] - metrics_std['Specificity'],
                     metrics_mean['Specificity'] + metrics_std['Specificity'],
                     alpha=0.2)

    plt.plot(time_steps, metrics_mean['Recall'], label='Sensitivity', marker='s')
    plt.fill_between(time_steps,
                     metrics_mean['Recall'] - metrics_std['Recall'],
                     metrics_mean['Recall'] + metrics_std['Recall'],
                     alpha=0.2)

    plt.plot(time_steps, metrics_mean['ROC AUC'], label='ROC AUC', marker='^')
    plt.fill_between(time_steps,
                     metrics_mean['ROC AUC'] - metrics_std['ROC AUC'],
                     metrics_mean['ROC AUC'] + metrics_std['ROC AUC'],
                     alpha=0.2)

    plt.plot(time_steps, metrics_mean['F1 Score'], label='F1 Score', marker='d', linestyle='--')
    plt.fill_between(time_steps,
                     metrics_mean['F1 Score'] - metrics_std['F1 Score'],
                     metrics_mean['F1 Score'] + metrics_std['F1 Score'],
                     alpha=0.2)

    plt.plot(time_steps, metrics_mean['Accuracy'], label='Accuracy', marker='x')
    plt.fill_between(time_steps,
                     metrics_mean['Accuracy'] - metrics_std['Accuracy'],
                     metrics_mean['Accuracy'] + metrics_std['Accuracy'],
                     alpha=0.2)

    plt.legend(fontsize=25)
    plt.xticks(ticks=range(1, T + 1), fontsize=25)
    plt.yticks(fontsize=25)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(0.85, 1.0)  # Adjust y-axis limits for new metrics

    plt.xlim(1, T)  # Adjust x-axis limits
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)

    if save_img:
        plt.tight_layout(pad=0)
        plt.savefig(path, bbox_inches='tight', pad_inches=0)

    plt.show()
    
    

def plot2_metrics_over_time(all_metrics, T, path, save_img=False):
    """
    Plot only Sensitivity (Recall), Specificity, and ROC AUC over time with shaded areas indicating standard deviation.
    Args:
        - all_metrics: A list with the temporal metrics of each split.
        - T: Number of time steps.
    """
    plt.figure(figsize=(12, 6))

    metrics_concat = pd.concat(all_metrics)
    metrics_mean = metrics_concat.groupby('Time Step').mean()
    metrics_std = metrics_concat.groupby('Time Step').std()

    time_steps = metrics_mean.index

    # Plot Specificity
    plt.plot(time_steps, metrics_mean['Specificity'], label='Specificity', marker='o', markersize=10) 
    plt.fill_between(time_steps,
                     metrics_mean['Specificity'] - metrics_std['Specificity'],
                     metrics_mean['Specificity'] + metrics_std['Specificity'],
                     alpha=0.2)

    # Plot Sensitivity (Recall)
    plt.plot(time_steps, metrics_mean['Recall'], label='Sensitivity', marker='s', markersize=10) 
    plt.fill_between(time_steps,
                     metrics_mean['Recall'] - metrics_std['Recall'],
                     metrics_mean['Recall'] + metrics_std['Recall'],
                     alpha=0.2)

    # Plot ROC AUC
    plt.plot(time_steps, metrics_mean['ROC AUC'], label='ROC AUC', marker='^', markersize=10)  
    plt.fill_between(time_steps,
                     metrics_mean['ROC AUC'] - metrics_std['ROC AUC'],
                     metrics_mean['ROC AUC'] + metrics_std['ROC AUC'],
                     alpha=0.2)

    # Customize plot appearance
    plt.legend(fontsize=25)
    plt.xticks(ticks=range(1, T + 1), fontsize=25)
    plt.yticks(fontsize=25)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(0.85, 1.0)  # Adjust y-axis limits

    plt.xlim(1, T)  # Adjust x-axis limits
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)

    if save_img:
        plt.tight_layout(pad=0)
        plt.savefig(path, bbox_inches='tight', pad_inches=0)

    plt.show()
    

###############################################
# FUNCTIONS FOR INTRINSIC EXPLAINABILITY #
###############################################
def plot_heatmap(attn_weights, features_list, X_test, split_num, patient_type):
    """
    Generates a heatmap visualization based on attention weights for a specific patient (AMR/NonAMR) and split number.
    
    Args:
        - attn_weights: numpy.ndarray. Attention weights for the input data.
        - features_list: List of feature names.
        - X_test: numpy.ndarray. Test data.
        - split_num: The split number. 
        - patient_type: "AMR" or "NonAMR"  
    """
    heatmap_att = np.zeros(attn_weights[0].shape)
    for j in range(attn_weights[0].shape[2]):
        heatmap_att[:, :, j] = np.where(X_test[:, :, j] == 666, np.nan, attn_weights[0][:, :, j])

    plt.figure(figsize=(18, 8))
    plt.subplot(2, 1, 1)
    heatmap = sns.heatmap(np.nanmean(heatmap_att, axis=0) / np.nanmean(heatmap_att, axis=0).max(), cmap='viridis')
    num_features = len(features_list)
    plt.xticks(np.arange(num_features) + 0.5, features_list, rotation=90)  # Center x-ticks

    
def plot_heatmap_av(attn_weights, features_list, X_test, patient_type):
    """
    Function to plot a heatmap based on attention weights.

    Args:
        - attn_weights: numpy.ndarray. Attention weights for the input data.
        - features_list: List of feature names.
        - X_test: numpy.ndarray. Test data.
        - split_num: The split number. 
        - patient_type: "AMR" or "NonAMR"  
    Returns: numpy.ndarray -> Normalized heatmap.
    """
    heatmap_att = np.zeros(X_test.shape)
    for j in range(attn_weights[0].shape[2]):
        heatmap_att[:, :, j] = np.where(X_test[:, :, j] == 666, np.nan, attn_weights[0][:, :, j])

    return np.nanmean(heatmap_att, axis=0) / np.nanmean(heatmap_att, axis=0).max()


def generate_heatmap(X_test, y_test, features_list, patient_type, n_time_steps=14):
    """
    Function to generate heatmap based on test data and labels.
    Args:
        - X_test: numpy.ndarray. Data for testing.
        - y_test: DataFrame with true labels.
        - features_list: List of feature names.
        - patient_type: "AMR" or "NonAMR" 
    Returns: Heatmaps for AMR and non-AMR patients.
    """
    index_AMR = []
    index_nonAMR = []

    y_test2D = y_test.loc[:, 'individualMRGerm'].values.flatten()

    for j in range(len(X_test)):
        y_patient_test = y_test2D[j * n_time_steps: (j + 1) * n_time_steps]
        if np.any(y_patient_test == 1):
            index_AMR.append(j)
        else:
            index_nonAMR.append(j)

    # Generate attention for AMR
    X_test_AMR = X_test[index_AMR]
    attn_weights_AMR = scores_layer([X_test_AMR])

    # Generate attention for non-AMR
    X_test_nonAMR = X_test[index_nonAMR]
    attn_weights_nonAMR = scores_layer([X_test_nonAMR])

    return plot_heatmap_av(attn_weights_AMR, features_list, X_test_AMR, patient_type), plot_heatmap_av(attn_weights_nonAMR, features_list, X_test_nonAMR, patient_type)


def average_heatmaps(heatmap_list):
    """
    Function to average a list of heatmaps.
    Args:
        - heatmap_list: List of heatmaps.
    Returns: numpy.ndarray -> Averaged heatmap.
    """
    return np.mean(heatmap_list, axis=0)