import multiprocessing as mp
import os
import sys
import time
import warnings
import pickle
import pandas as pd
import numpy as np

sys.path.append("../../../code")
from explainability_methods.pre_hoc import *
import pickle

warnings.filterwarnings("ignore")


def reset(T=8):
    f = ['vm1', 'vm3', 'vm4', 'vm5', 'vm13', 'vm20',
         'vm28', 'vm62', 'vm136', 'vm146', 'vm172', 'vm174', 'vm176', 'pm41',
         'pm42', 'pm43', 'pm44', 'pm87']

    tf = ['continua', 'continua', 'continua', 'continua', 'continua', 'continua', 'continua',
          'continua', 'continua', 'continua', 'continua', 'continua',
          'continua', 'continua', 'continua', 'continua',
          'continua', 'discreta']

    tf = tf * T

    return f, tf


def process_time_step(t, final_df, final_dl, params, F, T, weights):
    """
    Función para procesar un solo time step en paralelo.
    """
    init_T = time.time()

    X = final_df.iloc[:, : (t+1) * F].copy()
    y_day = final_dl.iloc[:, [t]]

    features, tipos_variables = reset()

    print(f"Processing t={t} | Samples x Features: {X.shape}")
    features = list(X.keys())

    indexesSelected = []
    MIvalues = []

    # Weights at time t
    weights_t = np.array(weights[:, t, :].flatten())

    for j in range((t+1) * F):
        try:
            if j == 0:
                X, z, featureSelected, maxMI = firstMI(X, y_day, params['k_n'], tipos_variables, params, weights_t)
            else:
                X, z, featureSelected, maxMI = myCondMI(X, y_day, z, params['k_n'], tipos_variables, params, weights_t)

            if not featureSelected or maxMI is None or np.isnan(maxMI) or np.isinf(maxMI):
                print(featureSelected, "-", maxMI)
                print(f"[WARNING] Invalid MI value at j={j}, t={t}. Skipping...")
                continue

            idx = features.index(featureSelected)
            del features[idx]
            del tipos_variables[idx]

            indexesSelected.append(featureSelected)
            MIvalues.append(maxMI)

        except Exception as e:
            print(f"[ERROR] Exception at j={j}, t={t}: {str(e)}")
            continue

    
    if len(MIvalues) > 0:
        normalization_factor = 1 / (T - t + 1)
        MIvalues = [mi * normalization_factor for mi in MIvalues]

    end_T = time.time()
    print(f"Time taken for t={t}: {end_T - init_T}")
    
    return indexesSelected, MIvalues


if __name__ == "__main__":

    #################
    dataset = 'CIRCULATORY'
    #################

    split_directory = './Results-App1/'
    params = {
        'k_n': 1,
        'intens': 1e-9,
        'val': 35,
        'mask_value': 666,
        'adjustment_factor': 1
    }
    norm = '0robustNorm'

    results = {}

    for split_num in [1, 2, 3]:
        init_split = time.time()

        xtr_path = f"../../../../DATA/{dataset}/s{split_num}/X_train_tensor_{norm}.npy"
        ytr_path = f"../../../../DATA/{dataset}/s{split_num}/y_train_tensor_{norm}.csv"

        features, tipos_variables = reset()
        final_df, final_dl, T, F = prepare_pop(split_num, features, norm, xtr_path, ytr_path)

        
        y = pd.read_csv(ytr_path)
        weights = create_temp_weight_mod(y, params, T)

        indexesSelected = []
        MIvalues = []

        if T > 0:
            with mp.Pool(processes=mp.cpu_count()) as pool:
                results_list = pool.starmap(
                    process_time_step,
                    [(t, final_df, final_dl, params, F, T, weights) for t in range(T)]
                )


            indexesSelected = [feature for sublist in results_list if sublist for feature in sublist[0]]
            MIvalues = [value for sublist in results_list if sublist for value in sublist[1]]

        results_df = pd.DataFrame({
            'Feature': indexesSelected,
            'MI Value': MIvalues
        })

        end_split = time.time()
        print(f"Tiempo total por split {split_num}:", end_split - init_split)

        results[f'results_df_{split_num}'] = results_df

    with open(os.path.join(split_directory, f"CCMI_results_population.pkl"), 'wb') as f:
        pickle.dump(results, f)
