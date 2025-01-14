import pickle  # Para cargar y guardar datos en formato pickle
import pandas as pd  # Para manejar estructuras de datos como DataFrames
import numpy as np  # Para manipulación de arrays numéricos
import matplotlib.pyplot as plt  # Para generar gráficos
import seaborn as sns  # Para generar heatmaps y mejorar gráficos
from mpl_toolkits.axes_grid1 import make_axes_locatable  # Para crear barras de color personalizadas en gráficos


def load_from_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def save_to_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def plot_shap_normalized(results, features, reordered_indices, final_order, title, time, save_img=False):
    """
    Heatmap plot based on SHAP values from 'results', reordered
    according to 'reordered_indices', with normalization to [-1, 1].
    
    - results: List of pandas DataFrames containing SHAP values.
    - reordered_indices: List of indices to reorder the data.
    - title: str title to save the plot.
    - time: int number of time steps.
    - save_img: bool, whether to save the plot as an image file.
    """
    df_conc = pd.concat(results)

    # Compute mean SHAP values across patients for each feature
    df_mean_pats = pd.DataFrame(columns=np.arange(1, time + 1, 1))
    for i in range(len(features)):
        df_mean_pats.loc[i] = df_conc[df_conc.index == features[i]].mean()

    # Order the time steps and the features 
    cols = df_mean_pats.columns.tolist()
    time_order = cols[::-1]
    df_final = df_mean_pats[time_order]
    df_final.columns = np.arange(0, time, 1)

    reordered_data = df_final.iloc[reordered_indices, :]

    # Normalize to [-1, 1]
    min_val = reordered_data.values.min()
    max_val = reordered_data.values.max()
    normalized_data = (2 * (reordered_data - min_val) / (max_val - min_val)) - 1

    # Heatmap
    plt.figure(figsize=(11, 22))
    ax = plt.gca()
    heatmap = sns.heatmap(
        normalized_data, 
        cmap="viridis", 
        xticklabels=np.arange(time) + 1,
        yticklabels=final_order,
        ax=ax,
        cbar=False  # Disable the default color bar
    )
    
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=25)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=25, rotation=90)
    
    # Create colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = heatmap.figure.colorbar(heatmap.collections[0], cax=cax)
    cbar.ax.tick_params(labelsize=25)
    
    num_ticks = 9
    tick_locs = np.linspace(-1, 1, num_ticks)
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels([f'{tick:.2f}' for tick in tick_locs])
    cbar.ax.tick_params(labelsize=25)

    plt.tight_layout(pad=0)
    if save_img:
        plt.savefig(title, bbox_inches='tight', pad_inches=0) 

    plt.show()
    
    return normalized_data


        
def plot_shap(results, features, reordered_indices, final_order, title, time, save_img=False):
    """
    Heatmap plot based on SHAP values from 'results', reordered
    according to 'reordered_indices', saved as .pdf with the name indicated in 'title'

    - results: List of pandas DataFrames containing SHAP values.
    - reordered_indices: List of indices to reorder the data.
    - title: str title to save the plot.
    
    """
    df_conc = pd.concat(results)

    df_mean_pats = pd.DataFrame(columns=np.arange(1,time+1,1))
    for i in range(len(features)):
        df_mean_pats.loc[i] = df_conc[df_conc.index == features[i]].mean()

    # Order the time steps and the features 
    cols = df_mean_pats.columns.tolist()
    time_order = cols[::-1]
    df_final = df_mean_pats[time_order]
    df_final.columns = np.arange(0,time,1)

    reordered_data = df_final.iloc[reordered_indices, :]
    data_max = reordered_data.values.max()
    data_min = reordered_data.values.min()
    
    # Heatmap
    plt.figure(figsize=(11,22))
    ax = plt.gca()
    heatmap = sns.heatmap(
        reordered_data, 
        cmap="viridis", 
        xticklabels=np.arange(time) + 1,
        yticklabels=final_order,
        ax=ax,
        cbar=False  # Disable the default color bar
    )
    
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=22)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=22, rotation=90)
    
    # Create colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = heatmap.figure.colorbar(heatmap.collections[0], cax=cax)
    cbar.ax.tick_params(labelsize=22)
    
    num_ticks = 9
    tick_locs = np.linspace(data_min, data_max, num_ticks)
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels([f'{tick:.3f}' for tick in tick_locs])
    cbar.ax.tick_params(labelsize=22)

    plt.tight_layout(pad=0)
    if save_img == True:
        plt.savefig(title, bbox_inches='tight', pad_inches=0) 

    plt.show()
    
    return reordered_data

def plot_shap_norm(matrix_amr, matrix_noamr, reordered_indices, final_order, title, time, save_img=False): 
    """
    Heatmap plot based on SHAP values from 'results', normalized to have the indices of AMR, reordered
    according to 'reordered_indices', saved as .pdf with the name indicated in 'title'    
    """

    min_amr, max_amr = matrix_amr.min().min(), matrix_amr.max().max()
    min_noamr, max_noamr = matrix_noamr.min().min(), matrix_noamr.max().max()
    
    matrix_noamr_transformed = (matrix_noamr - min_noamr) / (max_noamr - min_noamr) * (max_amr - min_amr) + min_amr
    data_max = matrix_noamr_transformed.values.max()
    data_min = matrix_noamr_transformed.values.min()

    plt.figure(figsize=(11,22))
    ax = plt.gca()
    heatmap = sns.heatmap(
        matrix_noamr_transformed, 
        cmap="viridis", 
        xticklabels=np.arange(time) + 1,
        yticklabels=final_order,
        ax=ax,
        cbar=False  # Disable the default color bar
    )
    
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=22)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=22, rotation=90)

    # Create colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = heatmap.figure.colorbar(heatmap.collections[0], cax=cax)
    cbar.ax.tick_params(labelsize=22)
    
    num_ticks = 9
    tick_locs = np.linspace(data_min, data_max, num_ticks)
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels([f'{tick:.2f}' for tick in tick_locs])
    cbar.ax.tick_params(labelsize=22)

    plt.tight_layout(pad=0)
    if save_img == True:
        plt.savefig(title, bbox_inches='tight', pad_inches=0) 

    plt.show()