import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
#
# def cosine_distance(a, b):


def clark_distance(a, b, eps=1e-10):
    """

    ref.: A. Levy, B. R. Shalom, and M. Chalamish, “A Guide to Similarity Measures,”
          Aug. 07, 2024, arXiv: arXiv:2408.07706. doi: 10.48550/arXiv.2408.07706.
    """
    # none-negatives
    a=np.abs(a)
    b=np.abs(b)
    return np.sqrt( np.sum(  (a-b)**2 / ( (a+b)**2  + eps)   ) )

def to_array(X, Y):
    if isinstance(X, dict):
        keys = sorted(X.keys())
        X_array = np.array([np.abs(X[k]["value"]) for k in keys]).reshape(1, -1)
    elif isinstance(X, list):
        keys = sorted(X[0]["stats"].keys())
        X_array = np.array([[np.abs(Xi["stats"][k]["value"]) for k in keys] for Xi in X])
    else:
        raise ValueError("X should be a generated sample dictionary or a list of samples")

    if isinstance(Y, dict):
        keys = sorted(Y.keys())
        Y_array = np.array([np.abs(Y[k]["value"]) for k in keys]).reshape(1, -1)
    elif isinstance(Y, list):
        keys = sorted(Y[0]["stats"].keys())
        Y_array = np.array([[np.abs(Yi["stats"][k]["value"]) for k in keys] for Yi in Y])
    else:
        raise ValueError("Y should be a generated sample dictionary or a list of samples")
    return X_array, Y_array

def to_matrix(X, Y):
    X_array, Y_array = to_array(X,Y)
    # Vectorized chi-square distance computation
    X_matrix = X_array[:, np.newaxis, :]  # Shape: (m, 1, d)
    Y_matrix = Y_array[np.newaxis, :, :]  # Shape: (1, n, d)
    return X_matrix, Y_matrix

def clark_distance_for_check(X, Y, epsilon=1e-10):
    """
    Compute the average chi-square distance between X and Y using vectorized operations.

    :param X: a generated sample dictionary, or a list of samples
    :param Y: a generated sample dictionary, or a list of samples
    :param epsilon: small constant to avoid division by zero
    :return: average chi-square distance
    """
    X_matrix, Y_matrix = to_matrix(X, Y)

    numerator = (X_matrix - Y_matrix) ** 2
    denominator = (X_matrix + Y_matrix)**2 + epsilon
    chi_sq = np.sqrt( np.sum(numerator / denominator, axis=2) ) # Shape: (m, n)

    return np.mean(chi_sq)