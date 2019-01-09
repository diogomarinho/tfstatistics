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
        return np.linalg.inv(l.T).dot(z)

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
        return np.linalg.inv(r).dot(q.T.dot(Y))

# lazy solution =P
class bySVD(leastSquaresSolver):
    def solve(self, X, Y):
        # solves the pseudo inverse which is the whole point of use SVD to compute least squares
        x_plus = np.linalg.pinv(X)
        return(x_plus.dot(Y))
