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
    print(y_train_amr)
    
    dfs = []
    for t in range(T):
        print(t)
        temp_df = pd.DataFrame(X_train_amr[:, t, :], columns=[f'{feature}_{t}' for feature in features])
        dfs.append(temp_df)
    final_df = pd.concat(dfs, axis=1)
    print(final_df)
    
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
    """
    Count the number of neighbors within a given radius for each point.

    Args:
        tree (BallTree or KDTree): Precomputed tree structure.
        x (np.array): Points for which neighbors are counted.
        r (float): Radius for the neighborhood.

    Returns:
        np.array: Number of neighbors within the radius for each point.
    """
   
    return tree.query_radius(x, r, count_only=True)


def add_noise(x, intens):
    """
    Add small noise to the input to break degeneracy.

    Args:
        x (np.array): Input array.
        intens (float): Intensity of the noise.

    Returns:
        np.array: Noisy input.
    """
    # small noise to break degeneracy, see doc.
    return x + intens * np.random.random_sample(x.shape)



def build_tree(points, val):
    """
    Build a tree structure for nearest neighbor queries.

    Args:
        points (np.array): Input points for the tree.
        val (int): Threshold to determine tree type.

    Returns:
        sklearn.neighbors.KDTree or BallTree: Tree structure.
    """
    if points.shape[1] >= val:
        return BallTree(points, metric='chebyshev')
    return KDTree(points, metric='chebyshev')



def query_neighbors(tree, x, k):
    """
    Query the k-th nearest neighbor distance.

    Args:
        tree (BallTree or KDTree): Tree structure.
        x (np.array): Points to query.
        k (int): Number of neighbors.

    Returns:
        np.array: Distance to the k-th neighbor.
    """
    return tree.query(x, k=k + 1)[0][:, k]



def avgdigamma(points, params, dvec):
    """
    Compute the average digamma value for a set of points.

    Args:
        points (np.array): Input points.
        dvec (np.array): Distances to neighbors.

    Returns:
        float: Average digamma value.
    """
    # This part finds number of neighbors in some radius in the marginal space
    # returns expectation value of <psi(nx)>
    tree = build_tree(points, params['val'])
    dvec = dvec - 1e-15
    num_points = count_neighbors(tree, points, dvec)
    return np.mean(digamma(num_points))

###########################################################
## END UTILITY FUNCTIONS ENTROPY
###########################################################

### FUNCT D-D (MI y CMI) ###
def entropyd(sx, base=2):
    """
    Estimate discrete entropy.

    Args:
        sx (np.array): List of samples.
        base (int): Base of the logarithm.

    Returns:
        float: Estimated entropy.
    """
    unique, count = np.unique(sx, return_counts=True, axis=0)
    # Convert to float as otherwise integer division results in all 0 for proba.
    proba = count.astype(float) / len(sx)
    # Avoid 0 division; remove probabilities == 0.0 (removing them does not change the entropy estimate as 0 * log(1/0) = 0.
    proba = proba[proba > 0.0]
    return np.sum(proba * np.log(1. / proba)) / log(base)

#### MI. D-D

def centropyd(x, y, base=2):
    """
    Estimate conditional entropy of X given Y.

    Args:
        x (np.array): Samples of X.
        y (np.array): Samples of Y.
        base (int): Base of the logarithm.

    Returns:
        float: Estimated conditional entropy.
    """
    xy = np.c_[x, y]
    return entropyd(xy, base) - entropyd(y, base)


def midd(x, y, base=2):
    """
    Estimate mutual information (MI) between discrete variables X and Y.

    Args:
        x (np.array): Samples of X.
        y (np.array): Samples of Y.
        base (int): Base of the logarithm.

    Returns:
        float: Estimated mutual information.
    """
    assert len(x) == len(y), "Arrays should have same length"
    return entropyd(x, base) - centropyd(x, y, base)


#### Cond. D-D
def cmidd(x, y, z, base=2):
    """
    Estimate conditional mutual information (CMI) between X and Y given Z.

    Args:
        x (np.array): Samples of X.
        y (np.array): Samples of Y.
        z (np.array): Samples of Z.
        base (int): Base of the logarithm.

    Returns:
        float: Estimated conditional mutual information.
    """
    assert len(x) == len(y) == len(z), "Arrays should have same length"
    xz = np.c_[x, z]
    yz = np.c_[y, z]
    xyz = np.c_[x, y, z]
    return entropyd(xz, base) + entropyd(yz, base) - entropyd(xyz, base) - entropyd(z, base)

#############################################################################

## MI - C-D
def entropy(x, k, params, base=2):
    """
    Estimate continuous entropy using k-nearest neighbors.

    Args:
        x (np.array): List of vectors.
        k (int): Number of nearest neighbors.
        base (int): Base of the logarithm.

    Returns:
        float: Estimated entropy.
    """

    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    x = np.asarray(x)
    n_elements, n_features = x.shape
    x = add_noise(x, params['intens'])
    tree = build_tree(x, params['val'])
    nn = query_neighbors(tree, x, k)
    const = digamma(n_elements) - digamma(k) + n_features * log(2)
    return (const + n_features * np.log(nn).mean()) / log(base)



def micd(x, y, k, params, base=2, warning=True):
    """
    Estimate mutual information (MI) between continuous X and discrete Y.

    Args:
        x (np.array): Continuous samples of X.
        y (np.array): Discrete samples of Y.
        k (int): Number of nearest neighbors.
        base (int): Base of the logarithm.
        warning (bool): Whether to display warnings for insufficient data.

    Returns:
        float: Estimated mutual information.
    """
    assert len(x) == len(y), "Arrays should have same length"
    entropy_x = entropy(x, k, params, base)
    y_unique, y_count = np.unique(y, return_counts=True, axis=0)
    y_proba = y_count / len(y)

    entropy_x_given_y = 0.
    for yval, py in zip(y_unique, y_proba):
        x_given_y = x[(y == yval).all(axis=1)]
        
        if k <= len(x_given_y) - 1:
            entropy_x_given_y += py * entropy(x_given_y, k, params, base)
        else:
            if warning:
                warnings.warn("Warning, after conditioning, on y={yval} insufficient data. "
                              "Assuming maximal entropy in this case.".format(yval=yval))
            entropy_x_given_y += py * entropy_x
    return abs(entropy_x - entropy_x_given_y)  # units already applied


# def midc(x, y, k, params, base=2, warning=True):
#     """
#     Estimate mutual information between discrete X and continuous Y.

#     Args:
#         x (np.array): Discrete samples of X.
#         y (np.array): Continuous samples of Y.
#         k (int): Number of nearest neighbors.
#         base (int): Base of the logarithm.
#         warning (bool): Whether to display warnings for insufficient data.

#     Returns:
#         float: Estimated mutual information.
#     """
#     return micd(y, x, k, params, base, warning)

## END MI - C-D

## CMI - C-D
def mi(x, y, z, k, params, base=2, alpha=0):
    """
    Estimate mutual information of X and Y, conditioned on Z if provided.

    Args:
        x (np.array): Samples of X.
        y (np.array): Samples of Y.
        z (np.array): Optional samples of Z for conditional MI.
        k (int): Number of nearest neighbors.
        base (int): Base of the logarithm.
        alpha (float): Correction parameter for large neighborhoods.

    Returns:
        float: Estimated mutual information.
    """
    assert len(x) == len(y), "Arrays should have same length"
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    x, y = np.asarray(x), np.asarray(y)
    x, y = x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1)
    x = add_noise(x, params['intens'])
    y = add_noise(y, params['intens'])
    points = [x, y]
    if z is not None:
        z = np.asarray(z)
        z = z.reshape(z.shape[0], -1)
        points.append(z)
    points = np.hstack(points)
    # Find nearest neighbors in joint space, p=inf means max-norm
    tree = build_tree(points, params['val'])
    dvec = query_neighbors(tree, points, k)
    if z is None:
        a, b, c, d = avgdigamma(x, params, dvec), avgdigamma(
            y, params, dvec), digamma(k), digamma(len(x))
        if alpha > 0:
            d += lnc_correction(tree, points, k, alpha)
    else:
        xz = np.c_[x, z]
        yz = np.c_[y, z]
        a, b, c, d = avgdigamma(xz, params, dvec), avgdigamma(
            yz, params, dvec), avgdigamma(z, params, dvec), digamma(k)
    return (-a - b + c + d) / log(base)


def cmi(x, y, z, k, params, base=2):
    """
    Wrapper for conditional mutual information (CMI).

    Args:
        x (np.array): Samples of X.
        y (np.array): Samples of Y.
        z (np.array): Samples of Z.

    Returns:
        float: Estimated conditional mutual information.
    """
    return mi(x, y, z, k, params, base=base)
## END CMI - C-D

######################################## STEP 0 ########################################

def firstMI(X, y, k, variable_types, params, base=2):
    """
    Select the variable with the highest mutual information with Y.

    Args:
        X (pd.DataFrame): Features.
        y (pd.DataFrame): Labels.
        k (int): Number of nearest neighbors.
        variable_types (list): Type of each variable ('discreta' or continuous).
        base (int): Base of the logarithm.

    Returns:
        X (pd.DataFrame): Remaining features after selection.
        z (np.array): Selected feature.
        key (str): Name of the selected feature.
        maxMI (float): Value of the highest mutual information.
    """
    maxMI = 0
    indexMIMax = 0
    
    claves = X.columns
    for k in range(X.shape[1]):
        # Select column in y corresponding to the day of the feature in X (f.e: feature 'AMG_0' with label of day 0)
        y_col = y.iloc[:, k // len(variable_types)].values

        # Filter the values in X and y to avoid the 666
        mask = (X.iloc[:, k].values != 666) & (y_col != 666)
        X_filtered = X.iloc[:, k].values[mask].reshape(-1, 1)
        y_filtered = y_col[mask].reshape(-1, 1)
        
        if variable_types[k] == 'discreta':
            miValue = np.abs(midd(X_filtered, y_filtered, base=base))
        else:
            miValue = np.abs(micd(X_filtered, y_filtered, k, params, base=base))
        
        if miValue > maxMI:
            maxMI = miValue
            indexMIMax = k
            
    # Eliminate the first variable and add it to z
    key = X.columns[indexMIMax]
    z = X[key].values.reshape(-1, 1)
    X = X.drop(columns=[key])
    
    return X, z, key, maxMI

def myCondMI(X, y, z, k, variable_types, params, base=2):
    """
    Select the variable with the highest conditional mutual information given Z.

    Args:
        X (pd.DataFrame): Features.
        y (pd.DataFrame): Labels.
        z (np.array): Conditional variables.
        k (int): Number of nearest neighbors.
        variable_types (list): Type of each variable ('discreta' or continuous).
        base (int): Base of the logarithm.

    Returns:
        X (pd.DataFrame): Remaining features after selection.
        z (np.array): Updated conditional variables.
        key (str): Name of the selected feature.
        maxMI (float): Value of the highest mutual information.
    """
    maxMI = 0
    indexMIMax = 0
    claves = X.columns
    for f in range(X.shape[1]):
        # Select column in y corresponding to the day of the feature in X (f.e: feature 'AMG_0' with label of day 0)
        y_col = y.iloc[:, f // len(variable_types)].values
        # Filter the values in X and y to avoid the 666
        mask = (X.iloc[:, f].values != 666) & (y_col != 666)
        X_filtered = X.iloc[:, f].values[mask].reshape(-1, 1)
        y_filtered = y_col[mask].reshape(-1, 1)
        z_filtered = z[mask]
                
        if variable_types[f] == 'discreta':
            miValue = np.abs(cmidd(X_filtered, y_filtered, z_filtered, base=base))
        else:
            miValue = np.abs(cmi(X_filtered, y_filtered, z_filtered, k, params, base=base))
        
        if miValue > maxMI:
            maxMI = miValue
            indexMIMax = f
            
    # Eliminate the first variable and add it to z
    key = X.columns[indexMIMax]
    z = np.append(z, X[key].values.reshape(-1, 1), axis=1)
    X = X.drop(columns=[key])
    
    return X, z, key, maxMI