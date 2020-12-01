import numpy as np
from numpy.linalg import pinv, inv, norm
from scipy.linalg import eig

def make_data(n, p):

    W = np.random.rand(n, n)
    W = (W + W.T) / 2
    D = np.diag(np.sum(W, axis=0))
    L_orig = D - W 
    L_orig = L_orig / np.trace(L_orig) * n
    [Lam, U] = eig(L_orig)
    Lam, U = Lam.real, U.real

    mu = np.zeros((n))
    sigma = pinv(np.diag(Lam))
    h = np.random.multivariate_normal(mu, sigma, p).T

    sigma_eps = 1e-2
    Y = U @ h
    X = Y + np.random.multivariate_normal(np.zeros((n)), sigma_eps*np.eye(n), p).T
    
    return L_orig, X