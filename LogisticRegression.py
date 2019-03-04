#!/usr/bin/python
import pandas as pd
import numpy as np
import statsmodels.api as sm
# implementation of newthon's method to solve the MLE
# of y~Binominal(logit(x))

# .
def sigmoid(x):
    return (1.0/(1.0 + np.exp(-x)))

# .
def getAlpha(x, theta):
    return sigmoid(np.dot(theta.T, x.T)).reshape(-1)

# .
def gradient(A, alpha, y):
    # print('compute gradient')
    return A.T.dot(alpha-y)

# .
def hessian(A, B):
    # print('compute hessian')
    return -A.T.dot(B).dot(A)

# .
if __name__ == '__main__':
    print('sketch of logistic regression')
    '''
        The test will be with a subset of the iris dataset
        the formula will be:
    '''
    # .
    test_df = pd.read_csv('~/projects/tfstatistics/data_iris.csv')
    test_df.Species = pd.factorize(test_df.Species)[0]
    y = test_df.Species.values
    x = test_df['Sepal.Length'].values
    x = np.column_stack((np.ones(shape=(x.shape[0])), x))
    A = np.copy(x)
    # iterations :
    # theta = np.zeros(A.shape[1]).reshape(-1, 1)
    theta = np.random.rand(A.shape[1]).reshape(-1, 1)
    for i in range(20):
        alpha = getAlpha(A, theta)
        B = np.diag(alpha - (1-alpha))
        G = gradient(A, alpha, y)
        H = hessian(A, B)
        theta = theta - ((np.linalg.inv(H)).dot(G)).reshape(-1,1)
    # . . . . . compare with GLM function
    glm = sm.GLM(y, x, family=sm.families.Binomial())
    result = glm.fit(maxiter=10)
    print(result.params)
    print(theta.reshape(-1))
