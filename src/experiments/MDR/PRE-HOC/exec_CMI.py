import multiprocessing as mp
import warnings
import os
import sys
import time
import pickle
import pandas as pd
import numpy as np

sys.path.append("../../../code")
from explainability_methods.pre_hoc import *
import pickle

warnings.filterwarnings("ignore")


def reset(T=14):
    f = ['AMG', 'CAR', 'CF1', 'CF3', 'CF4', 'Others', 'GLI', 'LIN', 'LIP', 'MAC', 'NTI', 'OXA', 'PAP', 'PEN', 'POL',
         'QUI', 'SUL', 'hoursVM', 'acinet.$_{pc}$', 'enterobac.$_{pc}$', 'enteroc.$_{pc}$',
         'pseud.$_{pc}$', 'staph.$_{pc}$', 'others.$_{pc}$', 'hoursICU', '# pat_atb', '# pat_MR',
         'CAR.$_{n}$', 'PAP.$_{n}$', 'Falta.$_{n}$', 'QUI.$_{n}$',
         'OXA.$_{n}$', 'PEN.$_{n}$', 'CF3.$_{n}$', 'GLI.$_{n}$',
         'CF4.$_{n}$', 'SUL.$_{n}$', 'NTI.$_{n}$', 'LIN.$_{n}$',
         'AMG.$_{n}$', 'MAC.$_{n}$', 'CF1.$_{n}$', 'POL.$_{n}$',
         'LIP.$_{n}$', '# pat_ttl', 'posture.$_{change}$',
         'insulin', 'nutr_art', 'sedation', 'relax', 'hep_fail',
         'renal_fail', 'coag_fail', 'hemo_fail',
         'resp_fail', 'multi_fail', 'n_transf',
         'vasoactive.$_{drug}$', 'dosis_nems', 'hoursTracheo', 'hoursUlcer',
         'hoursHemo', 'C01 PIVC 1',
         'C01 PIVC 2', 'C02 CVC - YD',
         'C02 CVC - SD', 'C02 CVC - SI', 'C02 CVC - FD',
         'C02 CVC - YI', 'C02 CVC - FI', '# catheters']

    tf = ['discreta', 'discreta', 'discreta', 'discreta', 'discreta',
          'discreta', 'discreta', 'discreta', 'discreta', 'discreta',
          'discreta', 'discreta', 'discreta', 'discreta', 'discreta',
          'discreta', 'discreta', 'continua', 'discreta',
          'discreta', 'discreta', 'discreta', 'discreta', 'discreta', 'continua',
          'continua', 'continua', 'continua',
          'continua', 'continua', 'continua', 'continua',
          'continua', 'continua', 'continua', 'continua',
          'continua', 'continua', 'continua',
          'continua', 'continua', 'continua', 'continua',
          'continua', 'continua', 'discreta', 'discreta',
          'discreta', 'discreta', 'discreta', 'discreta', 'discreta',
          'discreta', 'discreta', 'discreta',
          'discreta', 'continua', 'discreta', 'continua',
          'continua', 'continua', 'continua',
          'continua', 'continua', 'continua', 'continua', 'continua',
          'continua', 'continua', 'continua', 'continua']

    tf = tf * T
    return f, tf


def process_time_step(t, final_df, final_dl, params, F, T, weights):
    """
    Procesa un solo time step en paralelo.
    """
    init_T = time.time()
    
    # Seleccionar solo las columnas hasta el paso t actual
    X = final_df.iloc[:, : (t+1) * F].copy()
    y_day = final_dl.iloc[:, [t]]

    features, tipos_variables = reset()

    print(f"Procesando t={t} | Samples x Features: {X.shape}")
    features = list(X.keys())

    indexesSelected_t = []
    MIvalues_t = []

    # Obtener los pesos correspondientes al tiempo t
    weights_t = np.array(weights[:, t, :].flatten())

    for j in range((t+1) * F):
        try:
            if j == 0:
                X, z, featureSelected, maxMI = firstMI(X, y_day, params['k_n'], tipos_variables, params, weights_t)
                maxMI = maxMI/(maxMI*10)
            else:
                X, z, featureSelected, maxMI = myCondMI(X, y_day, z, params['k_n'], tipos_variables, params, weights_t)

            if not featureSelected or maxMI is None or np.isnan(maxMI) or np.isinf(maxMI):
                print(f"[WARNING] MI inválido en j={j}, t={t}. Saltando...")
                continue

            idx = features.index(featureSelected)
            del features[idx]
            del tipos_variables[idx]

            indexesSelected_t.append(featureSelected)
            MIvalues_t.append(maxMI)

        except Exception as e:
            print(f"[ERROR] Excepción en j={j}, t={t}: {str(e)}")
            continue

    # Normalización de valores MI
    if len(MIvalues_t) > 0:
        normalization_factor = 1 / (T - t + 1)
        MIvalues_t = [mi * normalization_factor for mi in MIvalues_t]

    end_T = time.time()
    print(f"Tiempo para t={t}: {end_T - init_T}")

    return indexesSelected_t, MIvalues_t


if __name__ == "__main__":

    #################
    dataset = 'MDR'
    #################

    split_directory = './Results-App1/'
    params = {
        'k_n': 1,
        'intens': 1e-5,
        'val': 1000,
        'mask_value': 666,
        'adjustment_factor': 1
    }
    norm = '0robustNorm'

    results = {}

    split_num = 3
    init_split = time.time()

    xtr_path = f"../../../../DATA/{dataset}/s{split_num}/X_train_tensor_{norm}.npy"
    ytr_path = f"../../../../DATA/{dataset}/s{split_num}/y_train_tensor_{norm}.csv"

    features, tipos_variables = reset()
    final_df, final_dl, T, F = prepare_pop(split_num, features, norm, xtr_path, ytr_path)

    # Calcular los pesos antes del bucle de tiempo
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

        # Recoger resultados paralelos
        indexesSelected = [feature for sublist in results_list if sublist for feature in sublist[0]]
        MIvalues = [value for sublist in results_list if sublist for value in sublist[1]]

    results_df = pd.DataFrame({
        'Feature': indexesSelected,
        'MI Value': MIvalues
    })

    end_split = time.time()
    print("Tiempo total:", end_split - init_split)

    results[f'results_df_{split_num}'] = results_df

    with open(os.path.join(split_directory, f"CCMI_results_population_s{split_num}.pkl"), 'wb') as f:
        pickle.dump(results, f)
