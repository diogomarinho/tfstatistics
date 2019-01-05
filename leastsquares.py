import numpy as np
from abc import ABC, abstractmethod

class leastSquaresSolver(ABC):
    @abstractmethod
    def solve(self, X, Y):
        print('Solving least squares')
#
class byNormalEquations(leastSquaresSolver):
    def solve(self, X, Y):
        X_T = X.T
        super().solve(X, Y)
        Xprod = np.dot(X_T, X)
        Yprod = np.dot(X_T, Y)
        return np.dot(np.linalg.inv(Xprod), Yprod)
#
class byGradientDescent(leastSquaresSolver):
    def solve(self, X, Y):
        # todo
        super().solve(X,Y)