import numpy as np

class leastSquaresSolver:
    def solve(self, X, Y):
        print('Solving least squares')
#
class byInverse(leastSquaresSolver):
    def solve(self, X, Y):
        Xprod = np.dot(X.T, X)
        Yprod = np.dot(X.T, Y)
        return np.dot(np.linalg.inv(Xprod), Yprod)
#
class byGradientDescent(leastSquaresSolver):
    def solve(self, X, Y):
        # todo
        pass;