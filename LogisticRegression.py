#!/usr/bin/python
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
# .
# implementation of newthon's method to solve the MLE
# of y~Binominal(logit(x))
# . sigmoid link
# .


# cross entropy function
def cross_entropy(x, y, sigmoid_probs):
    # return -np.mean(residuals(x, y, sigmoid_probs))
    return np.sum(y * np.log(sigmoid_probs) + (1.0 - sigmoid_probs) * np.log(1.0 - sigmoid_probs))


# log likelihood
def log_likelihood(x, y, theta):
    p1 = x.dot(theta)
    p2 = y * p1
    return np.sum(-np.log(1 + np.exp(p1)) + p2)


# sigmoid function
def sigmoid(x):
    return (1.0/(1.0 + np.exp(-x)))


# . diagonal
def getAlpha(x, theta):
    return sigmoid(np.dot(theta.T, x.T)).reshape(-1)


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
    # .
    glm = sm.GLM(y, x, family=sm.families.Binomial())
    result = glm.fit()
    print('Stats model:')
    print(result.params)
    # my newthon's method
    # Y = np.copy(y)
    # X = np.copy(x)
    max_iter = 15
    # np.random.seed(2019)
    # theta = np.random.rand(X.shape[1])
    theta = np.zeros(x.shape[1])
    # .
    for i in range(max_iter):
        # . print('iteration: {}'.format(i))
        alpha = getAlpha(x, theta)
        G = gradient(x, alpha, y)
        S = np.diag(alpha * (1.0 - alpha))  # diag matrix: how to make sparse
        H = hessian(x, S)
        iH = np.linalg.inv(H)
        # .
        theta_opt = theta + iH.dot(G)
        error = max(abs((theta_opt - theta)/theta_opt))
        if (error <= 1e-10):
            print('IRLS converged at iteration {}'.format(i))
            theta = theta_opt
            break
        theta = theta_opt
    # standard errors are the quare root of the inverted hessian
    se = np.sqrt(np.diag(iH))
    zscores = theta/se
    pvalues = 2.0 * stats.norm.cdf(-np.abs(zscores))
    print(pvalues)
    # pvalues =  2 * pnorm(zscores)
    # print(zscores)


    '''
    print('Mine:')
    print(theta.reshape(-1))
    print(H)
    alpha = getAlpha(x, theta)
    ce = cross_entropy(x, y, alpha)
    print('Cross entropy: {}'.format(ce))
    ll = log_likelihood(x, y, theta)
    print('Log-likelihood: {}'.format(ll))
    '''
    # print(residuals(x, y, theta))
    #
    #
    # from scipy import stats
    # # TODO check sign, why minus?
    # chi2stat = -score.dot(np.linalg.solve(hessian, score[:, None]))
    # pval = stats.chi2.sf(chi2stat, k_constraints)
    # # return a stats results instance instead?  Contrast?
    # return chi2stat, pval, k_constraints


# .
