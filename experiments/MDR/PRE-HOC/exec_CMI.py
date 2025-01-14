import warnings
warnings.filterwarnings("ignore")

import os
import sys
sys.path.append("../../../")

from rnns_architectures.pre_hoc import *
import pickle

def reset():
    f =  ['AMG', 'CAR', 'CF1', 'CF3', 'CF4',
            'Others', 'GLI', 'LIN', 'LIP', 'MAC', 'NTI', 'OXA', 'PAP', 'PEN', 'POL',
            'QUI', 'SUL', 'hoursVM', 'acinet.$_{pc}$', 'enterobac.$_{pc}$', 'enteroc.$_{pc}$',
            'pseud.$_{pc}$', 'staph.$_{pc}$', 'others.$_{pc}$', 'hoursICU',
            '# pat_atb', '# pat_MR',
            'CAR.$_{n}$', 'PAP.$_{n}$', 'Falta.$_{n}$', 'QUI.$_{n}$',
            'OXA.$_{n}$', 'PEN.$_{n}$', 'CF3.$_{n}$', 'GLI.$_{n}$',
            'CF4.$_{n}$', 'SUL.$_{n}$', 'NTI.$_{n}$', 'LIN.$_{n}$',
            'AMG.$_{n}$', 'MAC.$_{n}$', 'CF1.$_{n}$', 'POL.$_{n}$',
            'LIP.$_{n}$', '# pat_ttl','posture.$_{change}$',
            'insulin', 'nutr_art', 'sedation', 'relax', 'hep_fail',
            'renal_fail', 'coag_fail', 'hemo_fail',
            'resp_fail', 'multi_fail', 'n_transf',
            'vasoactive.$_{drug}$', 'dosis_nems', 'hoursTracheo', 'hoursUlcer',
            'hoursHemo', 'C01 PIVC 1',
            'C01 PIVC 2', 'C02 CVC - YD',
            'C02 CVC - SD', 'C02 CVC - SI', 'C02 CVC - FD',
            'C02 CVC - YI', 'C02 CVC - FI', '# catheters']

    tf =  ['discreta', 'discreta', 'discreta', 'discreta', 'discreta', 
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
                        'continua', 'continua', 'continua','continua']

    return f, tf

if __name__ == "__main__":
    
    #################
    dataset = 'MDR'
    #################

    split_directory = './Results-App1/'
    params = {
        'k_n': 1,
        'intens': 1e-15,
        'val': 35,
    }
    norm = '0robustNorm'

    results_amr = {}
    for split_num in [1,2,3]:

        xtr_path = f"../../../DATA/{dataset}/s{split_num}/X_train_tensor_{norm}.npy"
        ytr_path = f"../../../DATA/{dataset}/s{split_num}/y_train_tensor_{norm}.csv"

        features, tipos_variables = reset()
        final_df, final_dl, T, F = prepare_amr(split_num, features, norm, xtr_path, ytr_path)
        indexesSelected = []
        MIvalues = []
        
        print(final_dl)

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
        'intens':  1e-12,
        'val': 32,
    }

    for split_num in [1,2,3]:

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
    intens = 1e-12
    val = 32
    for split_num in [1,2,3]:

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
