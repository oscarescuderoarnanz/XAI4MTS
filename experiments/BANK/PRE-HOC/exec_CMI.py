import warnings
warnings.filterwarnings("ignore")

import os
import sys
sys.path.append("../../../")

from rnns_architectures.pre_hoc import *
import pickle

def reset():
    f =  ['year', 'Current_Assets', 'COGS', 'Depreciation_Amortization', 'EBITDA',
          
          'Inventory', 'Net_Income', 'Receivables', 'Market_Value', 'Net_Sales',
          
          'Total_Assets', 'Long-term_Debt', 'EBIT', 'Gross_Profit',
          
          'Current_Liabilities', 'Retained_Earnings', 'Total_Revenue',
          
          'Total_Liabilities', 'Operating_Expenses']
    

    tf =  ['continua', 'continua', 'continua', 'continua', 'continua',
          'continua', 'continua', 'continua', 'continua', 'continua',
          'continua', 'continua', 'continua', 'continua',
          'continua', 'continua', 'continua', 
          'continua', 'continua']

    return f, tf

if __name__ == "__main__":
    
    #################
    dataset = 'BANK'
    #################

    split_directory = './Results-App1/'
    params = {
        'k_n': 1,
        'intens': 1e-12,
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
        'intens':  1e-10,
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
    params = {
        'k_n': 1,
        'intens': 1e-10,
        'val': 32
    }

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


