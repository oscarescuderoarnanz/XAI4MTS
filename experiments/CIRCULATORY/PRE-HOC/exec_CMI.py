import warnings
warnings.filterwarnings("ignore")

import os
import sys
sys.path.append("../../../")

from rnns_architectures.pre_hoc import *
import pickle

def reset():
    f =  ['vm1', 'vm3', 'vm4', 'vm5', 'vm13', 'vm20',
       'vm28', 'vm62', 'vm136', 'vm146', 'vm172', 'vm174', 'vm176', 'pm41',
       'pm42', 'pm43', 'pm44', 'pm87']


    tf =  ['continua', 'continua', 'continua', 'continua', 'continua', 'continua', 'continua',
           'continua', 'continua', 'continua', 'continua', 'continua', 
           'continua', 'continua', 'continua', 'continua',
           'continua',  'discreta']

    return f, tf

if __name__ == "__main__":
    
    #################
    dataset = 'CIRCULATORY'
    #################

    split_directory = './Results-App1/'
    params = {
        'k_n': 1,
        'intens': 1e-9,
        'val': 35,
    }
    norm = '0robustNorm'

    results_amr = {}
    for split_num in [1, 2, 3]:

        xtr_path = f"../../../DATA/{dataset}/s{split_num}/X_train_tensor_{norm}.npy"
        ytr_path = f"../../../DATA/{dataset}/s{split_num}/y_train_tensor_{norm}.csv"

        features, tipos_variables = reset()
        final_df, final_dl, T, F = prepare_amr(split_num, features, norm, xtr_path, ytr_path)
        indexesSelected = []
        MIvalues = []

        for t in range(T):
            features, tipos_variables = reset()
            X_day = final_df.iloc[:, t*F:(t+1)*F]
            y_day = final_dl.iloc[:, [t]]
#             print(y_day)

            for j in range(F):
                if j == 0:
                    X_day, z, featureSelected, maxMI = firstMI(X_day, y_day, params['k_n'], tipos_variables, params)
                    maxMI = maxMI/(maxMI*10)
                else:
                    X_day, z, featureSelected, maxMI = myCondMI(X_day, y_day, z, params['k_n'], tipos_variables, params)

                feat = featureSelected.split("_"+str(t))[0]
                idx = features.index(feat)
                del features[idx]
                del tipos_variables[idx]

                featureSelected = featureSelected.split("_"+str(t))[0]
                indexesSelected.append(featureSelected)
                MIvalues.append(maxMI)

        results_df = pd.DataFrame({
            'Feature': indexesSelected,
            'MI Value': MIvalues
        })

        results_amr[f'results_df_{split_num}'] = results_df


    with open(os.path.join(split_directory, f"CMI_results_mdr.pkl"), 'wb') as f:
        pickle.dump(results_amr, f)
   
       
    # No-MDR
    results_noamr = {}
    params = {
        'k_n': 1,
        'intens':  1e-9,
        'val': 32,
    }

    for split_num in [1, 2, 3]:

        xtr_path = f"../../../DATA/{dataset}/s{split_num}/X_train_tensor_{norm}.npy"
        ytr_path = f"../../../DATA/{dataset}/s{split_num}/y_train_tensor_{norm}.csv"

        features, tipos_variables = reset()
        final_df, final_dl, T, F = prepare_noamr(split_num, features, norm,  xtr_path, ytr_path)
        indexesSelected = []
        MIvalues = []

        for t in range(T):
            features, tipos_variables = reset()
            #print("=========================["+str(t)+"]================================================")
            X_day = final_df.iloc[:, t*F:(t+1)*F]
            y_day = final_dl.iloc[:, [t]]

            for j in range(F):
                if j == 0:
                    X_day, z, featureSelected, maxMI = firstMI(X_day, y_day, params['k_n'], tipos_variables, params)
                    maxMI = maxMI/(maxMI*10)
                else:
                    X_day, z, featureSelected, maxMI = myCondMI(X_day, y_day, z, params['k_n'], tipos_variables, params)

                feat = featureSelected.split("_"+str(t))[0]
                idx = features.index(feat)
                del features[idx]
                del tipos_variables[idx]

                featureSelected = featureSelected.split("_"+str(t))[0]
                indexesSelected.append(featureSelected)
                MIvalues.append(maxMI)

        results_df = pd.DataFrame({
            'Feature': indexesSelected,
            'MI Value': MIvalues
        })

        results_noamr[f'results_df_{split_num}'] = results_df


    with open(os.path.join(split_directory, f"CMI_results_nomdr.pkl"), 'wb') as f:
        pickle.dump(results_noamr, f)
        
        
    # Population
    results_pop = {}
    intens = 1e-7
    val = 35
    for split_num in [1, 2, 3]:

        xtr_path = f"../../../DATA/{dataset}/s{split_num}/X_train_tensor_{norm}.npy"
        ytr_path = f"../../../DATA/{dataset}/s{split_num}/y_train_tensor_{norm}.csv"

        features, tipos_variables = reset()
        final_df, final_dl, T, F = prepare_pop(split_num, features, norm, xtr_path, ytr_path)
        indexesSelected = []
        MIvalues = []

        for t in range(T):
            features, tipos_variables = reset()
            print("=========================["+str(t)+"]================================================")
            X_day = final_df.iloc[:, t*F:(t+1)*F]
            y_day = final_dl.iloc[:, [t]]

            for j in range(F):
                if j == 0:
                    X_day, z, featureSelected, maxMI = firstMI(X_day, y_day, params['k_n'], tipos_variables, params)
                    maxMI = maxMI/(maxMI*10)
                else:
                    X_day, z, featureSelected, maxMI = myCondMI(X_day, y_day, z, params['k_n'], tipos_variables, params)

                feat = featureSelected.split("_"+str(t))[0]
                idx = features.index(feat)
                del features[idx]
                del tipos_variables[idx]

                featureSelected = featureSelected.split("_"+str(t))[0]
                indexesSelected.append(featureSelected)
                MIvalues.append(maxMI)

        results_df = pd.DataFrame({
            'Feature': indexesSelected,
            'MI Value': MIvalues
        })

        results_pop[f'results_df_{split_num}'] = results_df

    with open(os.path.join(split_directory, f"CMI_results_population.pkl"), 'wb') as f:
        pickle.dump(results_pop, f)
