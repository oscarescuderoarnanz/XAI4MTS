import os
import sys
sys.path.append("../../../")

from rnns_architectures.intrinsec import *
from rnns_architectures.utils import *
from IT_SHAP import TFWrapper, XAI_utils, utils_visualizations_IT_SHAP
from IT_SHAP.IT_SHAP import local_report
from tensorflow.keras.models import load_model
import seaborn as sns
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pickle
import random, os, json
import time

import warnings
warnings.filterwarnings("ignore")

def get_features():
    features =  ['vm1', 'vm3', 'vm4', 'vm5', 'vm13', 'vm20',
       'vm28', 'vm62', 'vm136', 'vm146', 'vm172', 'vm174', 'vm176', 'pm41',
       'pm42', 'pm43', 'pm44', 'pm87']
    
    print("# of features: ", len(features))

    # Show the features in the plot with a clinical order
    new_order = ['vm1', 'vm3', 'vm4', 'vm5', 'vm13', 'vm20',
       'vm28', 'vm62', 'vm136', 'vm146', 'vm172', 'vm174', 'vm176', 'pm41',
       'pm42', 'pm43', 'pm44', 'pm87']

    feature_idx = {feature: idx for idx, feature in enumerate(features)}
    reordered_indices = [feature_idx[feature] for feature in new_order]
    
    return features, reordered_indices



if __name__ == "__main__":

    # ############################################################################
    # Parameteres to define
    directory = './Results_LSTM'

    split = "s3"

    model = load_model('./Results_LSTM/split_3/model_split_3.h5')
    numberOftrPat = 1000
    norm = "robustNorm"
    numberOfTimeSteps = 8
    nsamples = 3200
    dataset = 'CIRCULATORY'
    # ############################################################################

    features, reordered_indices = get_features()
    # Step 0. Create the wrapper
    model_wrapped = TFWrapper.KerasModelWrapper(model)
    f_hs = lambda x, y=None: model_wrapped.predict_last_hs(x, y)

    # # Step 1. Load data
    X_train = np.load(f"../../../DATA/{dataset}/{split}/X_train_tensor_0{norm}.npy")
    X_test = np.load(f"../../../DATA/{dataset}/{split}/X_test_tensor_{norm}.npy")
    y_test = pd.read_csv(f"../../../DATA/{dataset}/{split}/y_test_tensor_"+norm+".csv")

    # Step 2. Create dataframe of train data
    data = X_train[0:numberOftrPat]
    df = pd.DataFrame(data.reshape(-1, data.shape[-1]), columns=features)

    adb_column = []
    for i in range(int(df.shape[0]/numberOfTimeSteps)):
        adb_column.extend([f'adb_{i}'] * numberOfTimeSteps)
    df['adb'] = adb_column

    # Add the column timeStep
    time_step_column = []
    for i in range(int(df.shape[0]/numberOfTimeSteps)):
        time_step_column.extend(list(range(numberOfTimeSteps)))
    df['timestamp'] = time_step_column
    d_train_normalized = df.copy()


    # Step 3. Get average of event and sequence
    average_event = XAI_utils.calc_avg_event(d_train_normalized, numerical_feats=features)
    average_sequence = XAI_utils.calc_avg_sequence(d_train_normalized, numerical_feats=features, model_features=features, entity_col="adb")

    # Step 4. Running shap for X_test patients
    results_shap = []
    ign_pat = []
    for idx_pat in range(X_test.shape[0]):
        print("===>", idx_pat)
        X = X_test[idx_pat]
        rows_filtered = np.any(X == 666, axis=-1)
        if len(rows_filtered[rows_filtered == False]) == 1:
            ign_pat.append(idx_pat)
            continue

        pos_x_data = np.expand_dims(X, axis=0)
        X_aux = X[:,~rows_filtered[0], :]

        event_dict = {'rs': 25, 'nsamples': nsamples}
        feature_dict = {'rs': 25, 'nsamples': nsamples, 'feature_names': features, 'plot_features': None}
        cell_dict = {'rs': 25, 'nsamples': nsamples, 'top_x_feats': pos_x_data.shape[2], 'top_x_events': X_aux.shape[1]}

        # local report with numpy instance ~ average_event
        _, _, cell_levelr = local_report(f_hs, pos_x_data, None, event_dict, feature_dict, 
                                         cell_dict=cell_dict, entity_uuid="adb", entity_col='adb', baseline=average_event)

        # reformat the dataframe always in the same order
        df = cell_levelr.copy()

        # Process the 'Event' column to get just the numbers
        df['Event'] = df['Event'].str.extract(r'(\d+)').astype(int)

        # Create a pivot table for the heat map
        pivot_table = df.pivot_table(index='Feature', columns='Event', values='Shapley Value', fill_value=0)
        pivot_table = pivot_table.reset_index()
        pivot_table.index = pivot_table['Feature'].values
        pivot_table = pivot_table.drop(['Feature'], axis=1)

        indices_actuales = list(pivot_table.index)
        posicion_indices = {indice: posicion for posicion, indice in enumerate(indices_actuales)}
        posiciones_reordenadas = [posicion_indices[indice] for indice in features]
        df_final = pivot_table.reindex(index=[indices_actuales[posicion] for posicion in posiciones_reordenadas])
        results_shap.append(df_final)

    save_to_pickle(results_shap, os.path.join(directory, f"results_shap_{split}.pkl"))