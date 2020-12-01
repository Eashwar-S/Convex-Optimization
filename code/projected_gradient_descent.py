import cvxpy as cp
import numpy as np
from numpy.linalg import pinv, inv, norm
from scipy.linalg import eig

def project_L(L0):
    
    n = L0.shape[0]
    
    L = cp.Variable((n,n), symmetric=True)
    
    obj = cp.Minimize(0.5 * cp.norm(L - L0, p='fro')**2)
    
    constraints = [cp.trace(L) == n]
    constraints += [L >> 0]
    constraints += [L @ np.ones((n)) == np.zeros((n))]
    for i in range(n):
        for j in range(i+1,n):
            constraints += [L[i,j] <= 0]

    prob = cp.Problem(obj, constraints)
    
    p_star = prob.solve()
    L_proj = L.value
    return L_proj

def proj_grad_descent(X, alpha, beta, L_init, num_iters, eta_L, eta_Y):
    
    """
    gradients w.r.t. L and Y
    """
    
    def grad_L(L, Y):
        return alpha * (Y @ Y.T) + 2 * beta * L

    def grad_Y(L, Y):
        return alpha * ((L + L.T) @ Y) - 2 * (X - Y)
    
    
    """
    projected gradient descent algorithm
    """
    
    L_iter = L_init
    Y_iter = X
    obj_old_val = np.inf
    for idx_iter in range(num_iters):
        
        L_iter = L_iter - eta_L * grad_L(L_iter, Y_iter)
        L_iter = project_L(L_iter)
        
        Y_iter = Y_iter - eta_Y * grad_Y(L_iter, Y_iter)

        obj_val = norm(X - Y_iter, 'fro')**2 + alpha * np.trace(Y_iter.T @ L_iter @ Y_iter) + beta * (norm(L_iter, 'fro')**2);


        if np.abs(obj_old_val - obj_val) < 1e-4:
            break
        else:
            obj_old_val = obj_val
    
    print(f"final objective value = {obj_val}")

    return L_iter, Y_iter, obj_val