#!/usr/bin/python
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.sparse import dia_matrix
# import scipy as sp
# implementation of newthon's method to solve the MLE
# of y~Binominal(logit(x))
# . sigmoid link

def log_likelihood(x, y, theta):
    sigmoid_probs = getAlpha(x, theta)
    return np.sum(y * np.log(sigmoid_probs) + (1 - y) * np.log(1 - sigmoid_probs))

def sigmoid(x):
    return (1.0/(1.0 + np.exp(-x)))

# . diagonal
def getAlpha(x, theta):
    arr = []
    for e in x:
        arr.append(sigmoid(theta.T.dot(e)))
    return(np.array(arr))
    #return sigmoid(np.dot(theta.T, x.T)).reshape(-1)

# . Gradient function of the MLE
def gradient(A, alpha, y):
    # print('compute gradient')
    return A.T.dot(y - alpha)

# . Hessian function of the MLE
def hessian(A, B):
    # print('compute hessian')
    return (A.T.dot(B)).dot(A)

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

    glm = sm.GLM(y, x, family=sm.families.Binomial())
    result = glm.fit(maxiter=10)
    print('Stats model:')
    print(result.params)

    # my newthon's method
    Y = np.copy(y)
    X = np.copy(x)
    max_iter = 15
    np.random.seed(2019)
    n,p = X.shape
    W = dia_matrix((n,n))        # W is initialized
    beta = np.linalg.lstsq(X, Y.reshape(-1,1))[0].reshape(-1)
    # import pdb; pdb.set_trace()
    # IRLS
    for iter in range(max_iter):
        pi = sigmoid(beta.T.dot(X.T)).reshape(-1)                # Evaluate the probabilities
        # W.setdiag(pi * (1 - pi)) # Set the diagonal
        W = np.diag(pi * (1 - pi))
        # Updating beta
        H = X.T.dot(W).dot(X)
        beta_star = beta + np.linalg.inv(H).dot(X.T).dot(y - pi)
        # Check for convergence
        error = max(abs((beta_star - beta)/beta_star))
        if error < 1e-10:
            print("Convergence reached after",iter+1,"iterations")
            break
        # If the convergence criterium is not satisfied, continue
        beta = beta_star
    print("Maximum iteration reached without convergence")
    print('mine beta')
    print(beta)
