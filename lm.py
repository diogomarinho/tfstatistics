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
        # coefficients calculation
        self.coefs = lss.solve(self.x, self.y)
        # estimates
        yhat = self.x.dot(self.coefs)
        # residuals
        self.res = self.y - yhat
        # squared residuals
        self.rss = self.res.T.dot(self.res)
        # degree's of freedom n - (p + 1(intercept is already stacked))
        self.df = self.x.shape[0] - self.x.shape[1]
        # standard deviation
        self.sigma2 = self.rss/self.df
        # x standard deviation
        xmean = np.mean(self.x, axis=0)
        xsd = np.sum((self.x - xmean)**2, axis=0)
        # pvalues
        # pvals
        #confidence intervals
        return self.coefs
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
print('Coefficients: \n{}'.format(lreg.coefs))
print('min res: {MIN}; max res: {MAX}'.format(MIN=np.min(lreg.res), MAX=np.max(lreg.res)))

#
# .
# . .
# . . .
# . . . . lreg.fit(nr)
#print('Coefficients: \n{}'.format(lreg.coefs))

# lreg.fit(svd)
#print('Coefficients: \n{}'.format(lreg.coefs))


