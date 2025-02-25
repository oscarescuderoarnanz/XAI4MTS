import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree, KDTree
from scipy.special import digamma
import warnings
from math import log


def prepare_amr(split_num, features, norm, xtr_path, ytr_path):
    """
    Prepare data for patients with antimicrobial resistance (AMR).

    Args:
        split_num (int): Split number of the dataset.
        features (list): List of feature names.
        norm (str): Normalization type.

    Returns:
        final_df (pd.DataFrame): Dataframe with AMR features.
        final_dl (pd.DataFrame): Dataframe with AMR labels.
        T (int): Number of time steps.
        F (int): Number of features.
    """
    
    X_train = np.load(xtr_path)
    y_train = pd.read_csv(ytr_path)
    
    y_train_aux = y_train[y_train.individualMRGerm != 666].reset_index(drop=True)
    y_train_aux = y_train_aux.groupby(by="Admissiondboid").sum().reset_index()
    amr = y_train_aux[y_train_aux.individualMRGerm != 0].index
    
    X_train_amr = X_train[amr]
    P, T, F = X_train.shape
    y_train_values = y_train[['individualMRGerm']].values.flatten()
    y_train_amr = y_train_values.reshape((P, T))
    y_train_amr = y_train_amr[amr]
    
    dfs = []
    for t in range(T):
        temp_df = pd.DataFrame(X_train_amr[:, t, :], columns=[f'{feature}_{t}' for feature in features])
        dfs.append(temp_df)
    final_df = pd.concat(dfs, axis=1)
    
    dls = []
    for t in range(T):
        temp_df = pd.DataFrame(y_train_amr[:, t], columns=[t])
        dls.append(temp_df)
    final_dl = pd.concat(dls, axis=1)
    
    return final_df, final_dl, T, F



def prepare_noamr(split_num, features, norm, xtr_path, ytr_path):
    """
    Prepare data for patients without antimicrobial resistance (No-AMR).

    Args:
        split_num (int): Split number of the dataset.
        features (list): List of feature names.
        norm (str): Normalization type.

    Returns:
        final_df (pd.DataFrame): Dataframe with No-AMR features.
        final_dl (pd.DataFrame): Dataframe with No-AMR labels.
        T (int): Number of time steps.
        F (int): Number of features.
    """
    
    X_train = np.load(xtr_path)
    y_train = pd.read_csv(ytr_path)
    
    y_train_aux = y_train[y_train.individualMRGerm != 666].reset_index(drop=True)
    y_train_aux = y_train_aux.groupby(by="Admissiondboid").sum().reset_index()
    noamr = y_train_aux[y_train_aux.individualMRGerm == 0].index
    
    X_train_noamr = X_train[noamr]
    P, T, F = X_train.shape
    y_train_values = y_train[['individualMRGerm']].values.flatten()
    y_train_noamr = y_train_values.reshape((P, T))
    y_train_noamr = y_train_noamr[noamr]
    
    dfs = [] 
    for t in range(T):
        temp_df = pd.DataFrame(X_train_noamr[:, t, :], columns=[f'{feature}_{t}' for feature in features])
        dfs.append(temp_df)
    final_df = pd.concat(dfs, axis=1)
    
    dls = [] 
    for t in range(T):
        temp_df = pd.DataFrame(y_train_noamr[:, t], columns=[t])
        dls.append(temp_df)
    final_dl = pd.concat(dls, axis=1)
    
    return final_df, final_dl, T, F


def prepare_pop(split_num, features, norm, xtr_path, ytr_path):
    """
    Prepare data for the entire population (AMR and No-AMR).

    Args:
        split_num (int): Split number of the dataset.
        features (list): List of feature names.
        norm (str): Normalization type.

    Returns:
        final_df (pd.DataFrame): Dataframe with population features.
        final_dl (pd.DataFrame): Dataframe with population labels.
        T (int): Number of time steps.
        F (int): Number of features.
        
    """
    X_train = np.load(xtr_path)
    y_train = pd.read_csv(ytr_path)
    
    y_train_aux = y_train[y_train.individualMRGerm != 666].reset_index(drop=True)
    y_train_aux = y_train_aux.groupby(by="Admissiondboid").sum().reset_index()
    pop = y_train_aux[y_train_aux.individualMRGerm >= 0].index
    
    X_train_pop = X_train[pop]
    P, T, F = X_train.shape
    y_train_values = y_train[['individualMRGerm']].values.flatten()
    y_train_pop = y_train_values.reshape((P, T))
    y_train_pop = y_train_pop[pop]
    
    dfs = [] 
    for t in range(T):
        temp_df = pd.DataFrame(X_train_pop[:, t, :], columns=[f'{feature}_{t}' for feature in features])
        dfs.append(temp_df)
    final_df = pd.concat(dfs, axis=1)
    
    dls = [] 
    for t in range(T):
        temp_df = pd.DataFrame(y_train_pop[:, t], columns=[t])
        dls.append(temp_df)
    final_dl = pd.concat(dls, axis=1)
    
    return final_df, final_dl, T, F

############################################################
# UTILITY FUNCTIONS
###########################################################

def count_neighbors(tree, x, r):
    """ Count the number of neighbors within a given radius for each point. """
    return tree.query_radius(x, r, count_only=True)


def add_noise(x, intens):
    """ Add small noise to the input to break degeneracy. """
    return x + intens * np.random.random_sample(x.shape)


def build_tree(points, val):
    """ Build a tree structure for nearest neighbor queries. """
    if points.shape[1] >= val:
        return BallTree(points, metric='chebyshev')
    return KDTree(points, metric='chebyshev')


def query_neighbors(tree, x, k):
    """ Query the k-th nearest neighbor distance. """
    k = min(k, len(x) - 1)  # Evita que k sea mayor que el número de muestras
    nn_distances = tree.query(x, k=k + 1)[0][:, k]
    return np.clip(nn_distances, 1e-10, None)  # Evita distancias cero


def avgdigamma(points, params, dvec):
    """ Compute the average digamma value for a set of points. """
    tree = build_tree(points, params['val'])
    dvec = np.clip(dvec - 1e-15, 1e-10, None)
    num_points = count_neighbors(tree, points, dvec)
    return np.mean(digamma(num_points))


###########################################################
# DISCRETE ENTROPY & MUTUAL INFORMATION
###########################################################

def entropyd(sx, base=2, sample_weight=None):
    """ Estimate discrete entropy, considering sample weights. """
    unique, count = np.unique(sx, return_counts=True, axis=0)
    proba = count.astype(float) / len(sx)

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight).flatten()
        sx = np.asarray(sx).flatten()

        weighted_counts = np.array([np.sum(sample_weight[sx == u]) for u in unique])
        weighted_proba = weighted_counts / np.sum(sample_weight)
    else:
        weighted_proba = proba

    weighted_proba = np.clip(weighted_proba, 1e-10, None)  # Evita log(0)
    return np.sum(weighted_proba * np.log(1. / weighted_proba)) / np.log(base)


def centropyd(x, y, base=2, sample_weight=None):
    """ Estimate conditional entropy of X given Y. """
    xy = np.c_[x, y]
    return entropyd(xy, base, sample_weight) - entropyd(y, base, sample_weight)


def midd(x, y, base=2, sample_weight=None):
    """ Estimate mutual information (MI) between discrete variables X and Y. """
    return entropyd(x, base, sample_weight) - centropyd(x, y, base, sample_weight)


def cmidd(x, y, z, base=2, sample_weight=None):
    """ Estimate conditional mutual information (CMI) between X and Y given Z. """
    xz = np.c_[x, z]
    yz = np.c_[y, z]
    xyz = np.c_[x, y, z]
    return entropyd(xz, base, sample_weight) + entropyd(yz, base, sample_weight) - \
           entropyd(xyz, base, sample_weight) - entropyd(z, base, sample_weight)


###########################################################
# CONTINUOUS ENTROPY & MUTUAL INFORMATION
###########################################################

def entropy(x, k, params, base=2, sample_weight=None):
    """ Estimate continuous entropy using k-nearest neighbors. """
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    x = np.asarray(x)
    x = add_noise(x, params['intens'])
    tree = build_tree(x, params['val'])
    nn = query_neighbors(tree, x, k)

    const = digamma(len(x)) - digamma(k) + x.shape[1] * np.log(2)
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight).flatten()
        entropy_value = (const + x.shape[1] * np.average(np.log(nn), weights=sample_weight)) / np.log(base)
    else:
        entropy_value = (const + x.shape[1] * np.log(nn).mean()) / np.log(base)

    return max(0, entropy_value)  # Evita valores negativos


def micd(x, y, k, params, base=2, sample_weight=None):
    """ Estimate mutual information (MI) between continuous X and discrete Y. """
    assert len(x) == len(y), "Arrays should have the same length"
    k = min(k, len(x) - 1)

    entropy_x = entropy(x, k, params, base, sample_weight)
    y_unique, y_count = np.unique(y, return_counts=True, axis=0)
    y_proba = y_count / len(y)

    entropy_x_given_y = 0.
    for yval, py in zip(y_unique, y_proba):
        x_given_y = x[(y == yval).all(axis=1)]
        weights_given_y = sample_weight[(y == yval).all(axis=1)] if sample_weight is not None else None

        if len(x_given_y) > k:
            entropy_x_given_y += py * entropy(x_given_y, k, params, base, weights_given_y)
        else:
            entropy_x_given_y += py * entropy_x
    return max(0, abs(entropy_x - entropy_x_given_y))  # Evita valores negativos


###########################################################
# MUTUAL INFORMATION WITH CONDITIONING
###########################################################

def mi(x, y, z, k, params, base=2, alpha=0):
    """ Estimate mutual information of X and Y, conditioned on Z if provided. """
    assert len(x) == len(y), "Arrays should have same length"
    k = min(k, len(x) - 1)

    x, y = np.asarray(x), np.asarray(y)
    x, y = x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1)
    x = add_noise(x, params['intens'])
    y = add_noise(y, params['intens'])

    points = [x, y]
    if z is not None:
        z = np.asarray(z).reshape(z.shape[0], -1)
        points.append(z)

    points = np.hstack(points)
    tree = build_tree(points, params['val'])
    dvec = query_neighbors(tree, points, k)

    if z is None:
        a, b, c, d = avgdigamma(x, params, dvec), avgdigamma(y, params, dvec), digamma(k), digamma(len(x))
    else:
        xz, yz = np.c_[x, z], np.c_[y, z]
        a, b, c, d = avgdigamma(xz, params, dvec), avgdigamma(yz, params, dvec), avgdigamma(z, params, dvec), digamma(k)

    return max(0, -a - b + c + d) / np.log(base)


def cmi(x, y, z, k, params, base=2, sample_weight=None):
    """ Estimate conditional mutual information (CMI) between X and Y given Z. """
    return max(0, mi(x, y, z, k, params, base, sample_weight))



######################################## STEP 0 ########################################

def firstMI(X, y, k, variable_types, params, weights, base=2):
    """
    Select the variable with the highest mutual information with Y, considering class imbalance with weights.

    Args:
        X (pd.DataFrame): Features.
        y (pd.DataFrame): Labels.
        k (int): Number of nearest neighbors.
        variable_types (list): Type of each variable ('discreta' or 'continua').
        weights (np.array): Sample weights for handling class imbalance.
        base (int): Base of the logarithm.

    Returns:
        X (pd.DataFrame): Remaining features after selection.
        z (np.array): Selected feature.
        key (str): Name of the selected feature.
        maxMI (float): Value of the highest mutual information.
    """
    maxMI = -np.inf
    indexMIMax = None

    for f in range(X.shape[1]):
        y_col = y.iloc[:, 0].values  # Extrae la primera columna de y
        mask = (X.iloc[:, f].values != 666) & (y_col != 666)
        
        if np.sum(mask) == 0:  # Si no hay valores válidos, continuar
            continue
        
        X_filtered = X.iloc[:, f].values[mask].reshape(-1, 1)
        y_filtered = y_col[mask].reshape(-1, 1)
        weights_filtered = weights[mask].reshape(-1, 1)  

        if variable_types[f] == 'discreta':
            miValue = np.abs(midd(X_filtered, y_filtered, base=base, sample_weight=weights_filtered))
        else:
            miValue = np.abs(micd(X_filtered, y_filtered, k, params, base=base, sample_weight=weights_filtered))
        
        if np.isinf(miValue) or np.isnan(miValue):  # Validación extra
            continue
        
        if miValue > maxMI:
            maxMI = miValue
            indexMIMax = f

    if indexMIMax is None:
        print("[WARNING] No valid feature found for MI computation. Returning None.")
        return X, None, None, None

    key = X.columns[indexMIMax]
    z = X[key].values.reshape(-1, 1)
    X = X.drop(columns=[key])

    return X, z, key, maxMI


def myCondMI(X, y, z, k, variable_types, params, weights, base=2):
    """
    Select the variable with the highest conditional mutual information given Z, considering class imbalance.

    Args:
        X (pd.DataFrame): Features.
        y (pd.DataFrame): Labels.
        z (np.array): Conditional variables.
        k (int): Number of nearest neighbors.
        variable_types (list): Type of each variable ('discreta' or 'continua').
        weights (np.array): Sample weights for handling class imbalance.
        base (int): Base of the logarithm.

    Returns:
        X (pd.DataFrame): Remaining features after selection.
        z (np.array): Updated conditional variables.
        key (str): Name of the selected feature.
        maxMI (float): Value of the highest mutual information.
    """
    maxMI = -np.inf
    indexMIMax = None

    for f in range(X.shape[1]):
        y_col = y.iloc[:, 0].values  # Extrae la primera columna de y
        mask = (X.iloc[:, f].values != 666) & (y_col != 666)
        
        if np.sum(mask) == 0:  # Si no hay valores válidos, continuar
            continue
        
        X_filtered = X.iloc[:, f].values[mask].reshape(-1, 1)
        y_filtered = y_col[mask].reshape(-1, 1)
        z_filtered = z[mask]
        weights_filtered = weights[mask]

        if variable_types[f] == 'discreta':
            miValue = np.abs(cmidd(X_filtered, y_filtered, z_filtered, base=base, sample_weight=weights_filtered))
        else:
            miValue = np.abs(cmi(X_filtered, y_filtered, z_filtered, k, params, base=base, sample_weight=weights_filtered))
        
        if np.isinf(miValue) or np.isnan(miValue):  # Validación extra
            continue
        
        if miValue > maxMI:
            maxMI = miValue
            indexMIMax = f

    if indexMIMax is None:
        print("[WARNING] No valid feature found for CMI computation. Returning None.")
        return X, z, None, None

    key = X.columns[indexMIMax]
    z = np.append(z, X[key].values.reshape(-1, 1), axis=1)
    X = X.drop(columns=[key])

    return X, z, key, maxMI
