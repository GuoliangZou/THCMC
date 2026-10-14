import numpy as np

def SolutionToP19(q0, m=1):
    """
    Solve the optimization problem:
      min 1/2 * sum_v ||s - qv||^2
      subject to s >= 0, sum(s) = 1

    Parameters:
    q0 : numpy array
        Input array
    m : int, optional
        Scaling factor, default is 1

    Returns:
    x : numpy array
        Solution to the optimization problem
    ft : int
        Number of iterations taken
    """
    ft = 1
    n = len(q0)
    p0 = np.sum(q0) / m - np.mean(np.sum(q0)) / m + 1 / n
    vmin = np.min(p0)

    if vmin < 0:
        f = 1
        lambda_m = 0
        while abs(f) > 1e-10:
            v1 = lambda_m - p0
            posidx = v1 > 0
            npos = np.sum(posidx)
            g = npos / n - 1
            if g == 0:
                g = np.finfo(float).eps  # Machine epsilon to avoid division by zero

            f = np.sum(v1[posidx]) / n - lambda_m
            lambda_m -= f / g
            ft += 1

            if ft > 100:
                x = np.maximum(-v1, 0)
                break
        else:
            x = np.maximum(-v1, 0)
    else:
        x = p0

    return x, ft
