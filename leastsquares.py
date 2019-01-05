import numpy as np
from abc import ABC, abstractmethod

class leastSquaresSolver(ABC):
    @abstractmethod
    def solve(self, X, Y):
        print('Solving least squares')
    def residuals(self):
        pass
    @classmethod
    def pvals(self):
        return('TODO')
    def confints(self):
        return ('TODO')

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
        super().solve(X,Y)

class byQRFactorization(leastSquaresSolver):
    def solve(self, X, Y):
        #TODO
        return

class bySVD(leastSquaresSolver):
    def solve(self, X, Y):
        #TODO
        return
