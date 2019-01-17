import numpy as np
from abc import ABC, abstractmethod

# .
class leastSquaresSolver(ABC):
    # returns the coefficients and the product (X'X)^1
    @abstractmethod
    def solve(self, X, Y):
        print('Solving least squares')
        return None, None
# .
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

# . phillippe assignment
class byGradientDescent(leastSquaresSolver):
    def solve(self, X, Y):
        # todo
        return 'NOT IMPLEMENTED'

# . columns of X must be linearly independent
class byQRFactorization(leastSquaresSolver):
    def solve(self, X, Y):
        # QR factorization
        q, r = np.linalg.qr(X)
        # from stats model
        return (np.linalg.solve(r, q.T.dot(Y)), np.linalg.inv(np.dot(r.T, r)))
        # my old version
        #return (np.linalg.inv(r).dot(q.T.dot(Y)), np.linalg.inv(np.dot(r.T, r)))

# . SVD
class bySVD(leastSquaresSolver):
    def solve(self, X, Y):
        rcond = np.asarray(1e-15)
        u, s, vt = np.linalg.svd(X, full_matrices=False)
        # discard small singular values
        cutoff = rcond[..., np.newaxis] * np.amax(s, axis=-1, keepdims=True)
        large = s > cutoff
        s = np.divide(1, s, where=large, out=s)
        s[~large] = 0
        # pseudo-inverse
        res = np.dot(np.transpose(vt), np.multiply(s[:, np.core.newaxis], np.transpose(u)))
        # res = np.matmul(np.transpose(vt), np.multiply(s[..., np.newaxis], np.transpose(u)))
        return res.dot(Y),  np.dot(res, np.transpose(res))
#