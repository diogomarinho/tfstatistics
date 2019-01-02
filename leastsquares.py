import numpy as np

class leastSquaresSolver:
    def solve(self, X, Y):
        print('Solving least squares')

class byInverse(leastSquaresSolver):
    def solve(self, X, Y):
        print('testing abstraction')
        #inverse = np.dot(np.transpose(self.X), self.X)
        #return(np.dot(inverse, (np.dot(np.transpose(self.X), self.Y))))
