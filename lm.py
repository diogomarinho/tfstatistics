#!/usr/bin/python3
import pandas as pd
import numpy as np
import tensorflow as tf
from leastsquares import leastSquaresSolver, byNormalEquation, byQRFactorization, bySVD

np.set_printoptions(suppress=True)
# response variabl (Y): SalePrice
# predictors: LotArea (Numerical) LotFrontage (Numerical) Street(Categorical)
# example data to develop the linear regression method

#def getXYData(fin='~/tfstatistics/house_pricing_train.csv'):
def getXYData(fin='~/Desktop/test2.csv'):
    df = pd.read_csv(fin)
    xydata = df['SalePrice LotArea LotFrontage'.split()]
    xydata = xydata.dropna(axis=0)
    xydata = pd.DataFrame(xydata, dtype=np.float32)
    #xydata.Street = xydata.Street.factorize()[0]
    # xydata.Street.cat.codes.values
    return xydata
# .
# class for linear regression model .
class lm:
    def __init__(self, data, response_id):
        self.y = data[[response_id]].values
        self.x = data.drop([response_id], axis=1).values
        self.x = np.column_stack((self.x, np.ones(shape=(self.x.shape[0]))))
    # .
    def fit(self, lss):
        # . coefficients calculation
        self.coefs, self.corr = lss.solve(self.x, self.y)
        # . coefficients correlation
        #
        # . estimates
        yhat = self.x.dot(self.coefs)
        #
        # . residuals
        self.res = self.y - yhat
        #
        # . residual sum of     squares |(sum((yhat - y)**2)|
        self.rss = self.res.T.dot(self.res)
        #
        # . total sum of squares  |sum((y - yean)^2)|
        y_mean = np.mean(self.y)
        self.tss = np.sum((self.y - y_mean)**2)
        #
        # . regression sum squares
        self.regSS = self.tss - self.rss
        #
        # . r2 (r-squared)
        self.r2 = self.tss - self.rss
        #
        # . degree's of freedom n - (p + 1(intercept is already stacked))
        self.df = self.x.shape[0] - self.x.shape[1]
        #
        # . residual standard error
        se = np.sqrt(self.rss/self.df)
        # .
        # . #
        # self.sigma2 = self.rss/self.df
        # standard errors
        # xmean = np.mean(self.x, axis=0)
        # xsd = np.sum((self.x - xmean)**2, axis=0)
        # p-values
        # pvals
        #confidence intervals
    # .
# .
# example
df = getXYData()
# .
qr = byQRFactorization()
nr = byNormalEquation()
svd = bySVD()
# .
lreg = lm(df, 'SalePrice')
# . fitting multiple lienar regression:
lreg.fit(qr)
print(lreg.corr)
#
lreg.fit(nr)
print(lreg.corr)
#
lreg.fit(svd)
print(lreg.corr)

#print('Coefficients: \n{}'.format(lreg.coefs))
#print('min res: {MIN}; max res: {MAX}'.format(MIN=np.min(lreg.res), MAX=np.max(lreg.res)))
#lreg.fit(nr)
#lreg.fit(svd)
#lreg.fit(qr)
from scipy import stats
#x = lreg.x
#y = lreg.y

#slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

#
# .
# . .
# . . .
# . . . . lreg.fit(nr)
#print('Coefficients: \n{}'.format(lreg.coefs))

# lreg.fit(svd)
#print('Coefficients: \n{}'.format(lreg.coefs))
