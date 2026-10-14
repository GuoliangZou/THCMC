import numpy as np


def eig1(A, c=None, isMax=True, isSym=True):
    """
    Calculate the top eigenvalues and eigenvectors of matrix A.

    Parameters:
    A : numpy array
        Input square matrix.
    c : int, optional
        Number of eigenvalues and eigenvectors to return. Defaults to the size of A.
    isMax : bool, optional
        If True, return the largest eigenvalues; otherwise, return the smallest.
    isSym : bool, optional
        If True, treat A as symmetric.

    Returns:
    eigvec : numpy array
        Eigenvectors corresponding to the top c eigenvalues.
    eigval : numpy array
        Top c eigenvalues.
    eigval_full : numpy array
        All eigenvalues in sorted order.
    """
    n = A.shape[0]
    if c is None or c > n:
        c = n

    if isSym:
        A = np.maximum(A, A.T)

    # Compute eigenvalues and eigenvectors
    d, v = np.linalg.eigh(A)  # Use eigh for symmetric matrices

    # Sort eigenvalues and eigenvectors
    if isMax:
        idx = np.argsort(d)[::-1]  # Descending order
    else:
        idx = np.argsort(d)  # Ascending order

    # Select top c eigenvalues and corresponding eigenvectors
    idx1 = idx[:c]
    eigval = d[idx1]
    eigvec = v[:, idx1]

    # Full sorted eigenvalues
    eigval_full = d[idx]

    return eigvec, eigval, eigval_full
