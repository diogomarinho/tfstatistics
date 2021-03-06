#!/usr/bin/python3
import pandas as pd
import numpy as np
import tensorflow as tf
from leastsquares import leastSquaresSolver, byNormalEquation, \
    byQRFactorization, bySVD, TFNormalEquations
from scipy import stats


np.set_printoptions(suppress=True)
# response variabl (Y): SalePrice
# predictors: LotArea (Numerical) LotFrontage (Numerical) Street(Categorical)
# example data to develop the linear regression method


#def getXYData(fin='~/tfstatistics/house_pricing_train.csv'):
def getXYData(fin='~/projects/tfstatistics/test_data.csv'):
    df = pd.read_csv(fin)
    df = df.dropna(0)
    return df


# class for linear regression model .
class lm:
    #
    def __init__(self, data, response_id):
        self.y = data[[response_id]].values
        self.x = data.drop([response_id], axis=1).values
        self.x = np.column_stack((np.ones(shape=(self.x.shape[0])), self.x))

    # . perform the computation
    def fit(self, lss):
        # . coefficients calculation
        self.coefs, self.corr = lss.solve(self.x, self.y)
        # self.bse = np.sqrt(np.diag(self.corr))
        # print(self.corr)
        # . coeffi++cients correlation
        #
        # . estimates
        yhat = self.x.dot(self.coefs)
        #
        # . residuals
        self.res = self.y - yhat
        #
        # . residual sum of squares |(sum((yhat - y)**2)|
        self.rss = self.res.T.dot(self.res).reshape(-1)
        #
        # . degree's of freedom n - p - 1(intercept is already stacked))
        self.df = self.x.shape[0] - self.x.shape[1]
        #
        # . mean square error
        self.mse = self.rss/self.df
        #
        # . variance of coefficients
        # . which other ways can we compute the the standard errors
        # . have to check the book tomorrow!
        self.var_b = self.mse * (self.corr.diagonal())
        # . standard deviation (error) of coefficients
        self.sd_b = np.sqrt(self.var_b)
        # t-value of the coefficients
        self.ts_b = self.coefs.reshape(-1)/self.sd_b.reshape(-1)
        # computing p-values from the t-distribution
        # self.p_values =  [2*(1-stats.t.cdf(np.abs(i), self.df)) for i in self.ts_b]
        self.p_values =  2.0 * stats.norm.cdf(-np.abs(self.ts_b))
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
        # .
        # . residual standard error
        se = np.sqrt(self.rss/self.df)

if __name__ == '__main__':
    # example
    df = getXYData()
    # .
    qr = byQRFactorization()
    nr = byNormalEquation()
    svd = bySVD()
    # .
    lreg = lm(df, 'SalePrice')
    # import pdb; pdb.set_trace()
    lreg.fit(qr)
    print('pvaleus mine {}'.format(lreg.p_values[0]))
    # tf_neq = TFNormalEquations()
    # theta, covar_params = tf_neq.solve(lreg.x, lreg.y)
    # print(covar_params)
    # print(theta)
    import statsmodels.api as sm
    x = df['LotFrontage LotArea'.split()].values
    y = df.SalePrice.values
    x = sm.add_constant(x)
    mod = sm.GLM(y, x, family=sm.families.Gaussian()).fit()
    print('pvalues statsmodel: {}'.format(mod.pvalues[0]))
    import pdb; pdb.set_trace()
    '''
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    mod = smf.ols(formula='SalePrice ~ LotArea + LotFrontage', data=df)
    #mod = sm.OLS(lreg.y, lreg.x)
    # .
    res = mod.fit()
    print(res.pvalues)
    #
    print('------ tf-stats X statsmodel -----------')
    coefs = lreg.coefs.reshape(-1)
    pvals = lreg.p_values
    coefs_ = res.params
    pvals_ = res.pvalues
    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . s
    print('pvalues')
    print('{A:15.20f} {B:15.20f} -- {S}'.format(A=pvals[0], B=pvals_[0], S=pvals[0] == pvals_[0]))
    print('{A:15.20f} {B:15.20f} -- {S}'.format(A=pvals[1], B=pvals_[1], S=pvals[1] == pvals_[1])) # getts different after 16 digits after the decimal place ??
    print('{A:15.20f} {B:15.20f} -- {S}'.format(A=pvals[2], B=pvals_[2], S=pvals[2] == pvals_[2])) # getts different after 16 digits after the decimal place ??
    print('Coeffcients:')
    print('{A:15.8f} {B:15.8f} -- {S}'.format(A=coefs[0], B=coefs_[0], S=coefs[0] == coefs_[0]))
    print('{A:15.8f} {B:15.8f} -- {S}'.format(A=coefs[1], B=coefs_[1], S=coefs[1] == coefs_[1]))
    print('{A:15.8f} {B:15.8f} -- {S}'.format(A=coefs[2], B=coefs_[2], S=coefs[2] == coefs_[2]))
    #
    print('standard errors')
    bse = lreg.sd_b
    print('{A:15.8f} {B:15.8f} -- {S}'.format(A=bse[0], B=res.bse[0], S=bse[0] == res.bse[0]))
    print('{A:15.8f} {B:15.8f} -- {S}'.format(A=bse[1], B=res.bse[1], S=bse[1] == res.bse[1]))
    print('{A:15.8f} {B:15.8f} -- {S}'.format(A=bse[2], B=res.bse[2], S=bse[2] == res.bse[2]))
    #
    print('t-values')
    ts = lreg.ts_b
    print('{A:15.8f} {B:15.8f} -- {S}'.format(A=ts[0], B=res.tvalues[0], S=ts[0] == res.tvalues[0]))
    print('{A:15.8f} {B:15.8f} -- {S}'.format(A=ts[1], B=res.tvalues[1], S=ts[1] == res.tvalues[1]))
    print('{A:15.8f} {B:15.8f} -- {S}'.format(A=ts[2], B=res.tvalues[2], S=ts[2] == res.tvalues[2]))
    '''
