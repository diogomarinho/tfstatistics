import numpy as np
from abc import ABC, abstractmethod

class leastSquaresSolver(ABC):
    @abstractmethod
    def solve(self, X, Y):
        print('Solving least squares')
#
class byInverse(leastSquaresSolver):
    def solve(self, X, Y):
        super().solve(X, Y)
        Xprod = np.dot(X.T, X)
        Yprod = np.dot(X.T, Y)
        return np.dot(np.linalg.inv(Xprod), Yprod)
#
class byGradientDescent(leastSquaresSolver):
    def solve(self, X, Y):
        # todo
        super().solve(X,Y)