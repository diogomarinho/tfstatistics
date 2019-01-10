import numpy as np
from abc import ABC, abstractmethod

class leastSquaresSolver(ABC):
    @abstractmethod
    def solve(self, X, Y):
        print('Solving least squares')

class byNormalEquation(leastSquaresSolver):
    # this has the same effect of :
    # (X_T . X )' * (X_T . Y)
    def solve(self, X, Y):
        x_t = X.T
        xprod = x_t.dot(X)
        l = np.linalg.cholesky(xprod)
        d = x_t.dot(Y)
        # print(d.shape)
        # solve L . z = d  ---->  z = L' * d
        z = np.linalg.inv(l).dot(d)
        # solve L^t . x = z --->  x =  Lˆt' * z
        return (np.linalg.inv(l.T).dot(z), np.linalg.inv(l.dot(l.T)))

# phillippe assignment
class byGradientDescent(leastSquaresSolver):
    def solve(self, X, Y):
        # todo
        return 'NOT IMPLEMENTED'

# columns of X must be linearly independent
class byQRFactorization(leastSquaresSolver):
    def solve(self, X, Y):
        # QR factorization
        q, r = np.linalg.qr(X)
        # coefficients
        return (np.linalg.inv(r).dot(q.T.dot(Y)), np.linalg.inv(np.dot(r.T, r)))

# SVD
class bySVD(leastSquaresSolver):
    def solve(self, X, Y):
        u, s, v = np.linalg.svd(X, full_matrices=False)
        X_ = u.dot(np.diag(s)).dot(v)
        n = X.shape[1]
        r = np.linalg.matrix_rank(X)
        sigma_inv = np.diag(np.hstack([1 / s[:r], np.zeros(n - r)]))
        X_plus = v.dot(sigma_inv).dot(u.T)
        return(X_plus.dot(Y), np.linalg.inv(X_.T.dot(X_)))
