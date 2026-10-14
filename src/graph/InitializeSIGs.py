import numpy as np
from scipy.spatial.distance import cdist

def InitializeSIGs(X, k=15, issymmetric=1):
    """
    Initialize the similarity matrix S and distance matrix D.

    Parameters:
    X : numpy array of shape (n_samples, n_features)
        Each row is a data point.
    k : int
        Number of neighbors.
    issymmetric : int
        If 1, symmetrize the similarity matrix S.

    Returns:
    S : numpy array of shape (n_samples, n_samples)
        Similarity matrix.
    D : numpy array of shape (n_samples, n_samples)
        Pairwise squared Euclidean distance matrix.
    """
    n = X.shape[0]
    # Compute pairwise squared Euclidean distances
    D = cdist(X.cpu(), X.cpu(), 'sqeuclidean')
    # Sort each row
    idx = np.argsort(D, axis=1)

    S = np.zeros((n, n))
    eps = np.finfo(float).eps  # Machine epsilon to prevent division by zero

    for i in range(n):
        # Indices of the k+1 nearest neighbors (excluding the first which is the point itself)
        id = idx[i, 1:k+2]
        di = D[i, id]
        dk1 = di[k]  # Corresponds to di(k+1) in MATLAB
        numerator = dk1 - di[:k]
        denominator = k * dk1 - np.sum(di[:k]) + eps
        S[i, id[:k]] = numerator / denominator

    if issymmetric == 1:
        S = (S + S.T) / 2

    return S
