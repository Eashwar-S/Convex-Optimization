import cvxpy as cp
import numpy as np
from numpy.linalg import pinv, inv, norm
from scipy.linalg import eig

def logdet(X, lam):
    
    n, p = X.shape
        
    W = cp.Variable((n, n), symmetric=True)
    Lpre = cp.Variable((n, n), PSD=True)
    sigma_sqr = cp.Variable(pos=True)
    
    
    
    obj = cp.Minimize((1/p)*cp.trace(Lpre @ X @ X.T) - cp.atoms.log_det(Lpre) + (lam/p)*cp.norm(W, p=1))
        
    constraints = [Lpre == cp.diag(cp.atoms.affine.sum.sum(W, axis=0)) - W + np.eye(n)*sigma_sqr]
#     constraints += [cp.diag(W) == 0]
#     constraints += [cp.reshape(W, (n*n, 1)) >= 0]
    for i in range(0,n):
        for j in range(0,n):
            if i==j:
                constraints += [W[i,j] == 0]
            else:
                constraints += [W[i,j] >= 0]
                
    prob = cp.Problem(obj, constraints)
    
    p_star = prob.solve()
    L_opt = Lpre.value
    
    return L_opt
