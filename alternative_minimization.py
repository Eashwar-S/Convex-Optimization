import cvxpy as cp
import numpy as np
from numpy.linalg import pinv, inv, norm
from scipy.linalg import eig

def alternative_minimization(X, alpha, beta, num_iters, dual=False):
    
    n = X.shape[0]
    
    """
    helper functions
    """
    
    def dual_helper(Y):
        m = n * (n + 1) // 2

        # position matrix
        Q = np.zeros((n, n))
        count = 0
        for d in np.arange(0, n):
            for i in np.arange(d, n):
                j = i - d
                Q[i, j] = count
                count += 1
        for d in np.arange(1, n):
            for i in np.arange(0, n - d):
                j = i + d
                Q[i, j] = count
                count += 1

        # C = vec(Y @ Y.T)
        YYt = Y @ Y.T
        C = np.zeros(((n**2), 1))
        for i in np.arange(0, n):
            for j in np.arange(0, n):
                loc = int(Q[i, j])
                C[loc] = YYt[i, j] 

        # Duplication matrix M
        nsqr = n ** 2
        M1 = np.eye(m)
        M2 = np.hstack((np.zeros((nsqr - m, n)), np.eye(m - n)))
        M = np.vstack((M1, M2))

        # Equality constraints matrix A
        A1 = np.hstack((np.ones((1, n)), np.zeros((1, m - n))))
        A2 = np.hstack((np.eye(n), np.zeros((n, m - n))))
        for i in np.arange(0, n):
            row = i
            for j in np.arange(0, i):
                col = int(Q[i, j])
                A2[row, col] = 1
            for j in np.arange(i + 1, n):
                col = int(Q[j, i])
                A2[row, col] = 1
        A = np.vstack((A1, A2))

        # vector b
        b = np.vstack((n * np.ones((1, 1)), np.zeros((n, 1))))

        # Inequality constraints matrix G
        G = np.hstack((np.zeros((m - n, n)), np.eye(m - n)))

        # vector h
        h = np.zeros((m - n, 1))

        P = 2 * beta * M.T @ M
        q = (alpha * M.T @ C).squeeze()

        return Q, P, q, G, h, A, b

    def make_L(x, Q):
        L = np.zeros((n, n))
        for i in np.arange(0, n):
            for j in np.arange(0, i + 1):
                loc = int(Q[i, j])
                L[i, j] = x[loc]

        L = L + L.T - np.diag(np.diag(L))
        return L
    
    """
    QP optimizations for L and Y
    """
    
    def optimize_L_primal(Y):
        L = cp.Variable((n,n), symmetric=True)

        obj = cp.Minimize(alpha * cp.trace(Y.T @ L @ Y) + beta * (cp.norm(L, p='fro')**2))

        constraints = [cp.trace(L) == n]
        constraints += [L >> 0]
        constraints += [L @ np.ones((n)) == np.zeros((n))]
        for i in range(n):
            for j in range(i+1,n):
                constraints += [L[i,j] <= 0]

        prob = cp.Problem(obj, constraints)

        p_star = prob.solve()
        print(f"primal value is {p_star}")
        L_opt = L.value
        return L_opt

    def optimize_L_dual(Y):
        Q, P, q, G, h, A, b = dual_helper(Y)

        lam = cp.Variable((G.shape[0]))
        nu = cp.Variable((A.shape[0]))

        Pinv = pinv(P)
        r = q + G.T @ lam + A.T @ nu

        obj_dual = cp.Maximize((-0.5) * cp.quad_form(r, Pinv) - h.T @ lam - b.T @ nu)

        constraints_dual = [lam >= 0]

        prob_dual = cp.Problem(obj_dual, constraints_dual)

        p_dual_star = prob_dual.solve()
        print(f"dual value is {p_dual_star}")
        lam_opt, nu_opt = lam.value, nu.value

        x = - Pinv @ (r.value)
        L_opt = make_L(x, Q)
        return L_opt

    def optimize_Y(L):
        Y_opt = inv(np.eye(n) + alpha * L) @ X

        return Y_opt
    
    """
    alternate minimization algorithm
    """
    
    Y_iter = X
    obj_old_val = np.inf
    for idx_iter in np.arange(num_iters):
        
        print(f"iteration #{idx_iter}")
        
        if dual is True:
            L_iter = optimize_L_dual(Y_iter)
        else:
            L_iter = optimize_L_primal(Y_iter)
        
        Y_iter = optimize_Y(L_iter)
        
        obj_val = norm(X - Y_iter, 'fro')**2 + alpha * np.trace(Y_iter.T @ L_iter @ Y_iter) + beta * (norm(L_iter, 'fro')**2);

        print(f"iteration-{idx_iter}: obj value = {obj_val}")

        if np.abs(obj_old_val - obj_val) < 1e-4:
            break
        else:
            obj_old_val = obj_val

    return L_iter, Y_iter
