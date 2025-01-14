import warnings
warnings.filterwarnings("ignore")

import os
import sys
sys.path.append("../../../")

from rnns_architectures.pos_hoc import *
from rnns_architectures.utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pickle
import random, os, json
import time

if __name__ == "__main__":
    seeds = [9, 18, 35]

    ####################
    input_shape = 26
    n_time_steps = 14
    dataset = 'MDR/FS_CMI'
    ####################
    batch_size = 32
    n_epochs_max = 1000

    layer_list = [
        [input_shape, 3, 1], [input_shape, 5, 1], [input_shape, 10, 1],
        [input_shape, 20, 1],  [input_shape, 30, 1], [input_shape, 40, 1], 
        [input_shape, 50, 1], [input_shape, 60, 1]
    ]

    dropout = [0.0, 0.15, 0.3]
    lr_scheduler = [1e-1, 1e-2, 1e-3, 1e-4]

    adjustment_factor = [1]  


    activation = ['tanh', 'LeakyReLU']

    norm = "robustNorm"
    model_type = 'LSTM'
    hyperparameters = {
        "n_time_steps": n_time_steps,
        "mask_value": 666,
        "batch_size": batch_size,
        "n_epochs_max": n_epochs_max,
        "monitor": "val_loss",
        "mindelta": 0,
        "patience": 50,
        "dropout": 0.0,
        "verbose": 0,
        'dataset': dataset,
        'model_type': model_type
    }


    loss_train = []
    loss_dev = []
    v_models = []

    bestHyperparameters_bySplit = {}
    y_pred_by_split = {}

    for i in range(1,4):
        init = time.time()
        split = f"s{i}"
        paths = {
            'x_tr': f"../../../DATA/{dataset}/{split}/X_train_tensor_{i-1}{norm}.npy",
            'y_tr': f"../../../DATA/{dataset}/{split}/y_train_tensor_{i-1}{norm}.csv",
            'x_val': f"../../../DATA/{dataset}/{split}/X_val_tensor_{i-1}{norm}.npy",
            'y_val': f"../../../DATA/{dataset}/{split}/y_val_tensor_{i-1}{norm}.csv"
        }
        X_test = np.load(f"../../../DATA/{dataset}/s" + str(i) + "/X_test_tensor_" + norm + ".npy")
        y_test = pd.read_csv(f"../../../DATA/{dataset}/s" + str(i) + "/y_test_tensor_" + norm + ".csv")

        # GridSearch of hyperparameters 
        bestHyperparameters = myCVGridParallel(hyperparameters,
                                               dropout,
                                               lr_scheduler,
                                               layer_list,
                                               adjustment_factor,
                                               activation,
                                               seeds[i-1],
                                               split,
                                               norm,
                                               n_time_steps)
        fin = time.time()
        X_train = bestHyperparameters["X_train"]
        y_train = bestHyperparameters["y_train"]
        X_val = bestHyperparameters["X_val"]
        y_val = bestHyperparameters["y_val"]

        bestHyperparameters_bySplit[str(i)] = bestHyperparameters

        # Save best hyperparameters for current split
        split_directory = './Results_LSTM_CMI/split_' + str(i)
        if not os.path.exists(split_directory):
            os.makedirs(split_directory)

        with open(os.path.join(split_directory, f"bestHyperparameters_split_{i}.pkl"), 'wb') as f:
            pickle.dump(bestHyperparameters, f)

        hyperparameters = {
            'n_time_steps': hyperparameters["n_time_steps"],
            'mask_value': hyperparameters["mask_value"],
            'dataset': hyperparameters['dataset'],

            'batch_size': hyperparameters["batch_size"],
            'n_epochs_max': hyperparameters["n_epochs_max"],
            'monitor':  hyperparameters["monitor"],
            "mindelta": hyperparameters["mindelta"],
            "patience": hyperparameters["patience"],
            "dropout": bestHyperparameters["dropout"],
            "layers": bestHyperparameters["layers"],
            "lr_scheduler": bestHyperparameters["lr_scheduler"],
            "adjustment_factor": bestHyperparameters["adjustment_factor"],
            "activation": bestHyperparameters["activation"],
            'model_type': hyperparameters['model_type'],
            'verbose': 0
        }

        # --- TRY ON TEST -----------------------------------------------------------------------

        reset_keras()

        # Create temporal weights
        sample_weights_train = create_temp_weight_mod(y_train, hyperparameters, timeSteps=n_time_steps)
        sample_weights_val = create_temp_weight_mod(y_val, hyperparameters, timeSteps=n_time_steps)

        model, hist, early = run_network(
            X_train, X_val,
            y_train.loc[:, 'individualMRGerm'].values.reshape(y_train.shape[0] // n_time_steps, n_time_steps, 1),
            y_val.loc[:, 'individualMRGerm'].values.reshape(y_val.shape[0] // n_time_steps, n_time_steps, 1),
            sample_weights_train, sample_weights_val,
            hyperparameters,
            seeds[i-1]
        )

        v_models.append(model)
        loss_train.append(hist.history['loss'])
        loss_dev.append(hist.history['val_loss'])

        y_pred = model.predict(x=X_test)
        y_pred_by_split[str(i)] = y_pred

        # Save y_pred for current split
        with open(os.path.join(split_directory, f"y_pred_split_{i}.pkl"), 'wb') as f:
            pickle.dump(y_pred, f)

        # Save model for current split
        model_filename = os.path.join(split_directory, f"model_split_{i}.h5")
        model.save(model_filename)

    # END EXECUTION - SAVE AGGREGATED RESULTS

