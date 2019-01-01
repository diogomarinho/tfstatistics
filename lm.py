#!/usr/bin/python3
import pandas as pd
import tensorflow as tf

# response variabl (Y): SalePrice
# predictors: LotArea (Numerical) LotFrontage (Numerical) Street(Categorical)
# example data to develop the linear regression method
def getXYData(fin='~/tfstatistics/house_pricing_train.csv'):
    df = pd.read_csv(fin)
    xydata = df['SalePrice LotArea LotFrontage Street'.split()]
    xydata = xydata.dropna(axis=0)
    xydata.Street = xydata.Street.factorize()[0]
    return xydata

def lmfit(response=None, predictors=None)
    #TODO
    residuals  = None
    intercept = None
    coefs = None
    pvals = None
    conf_left = None
    conf_right = None
    return residuals, intercept, coefs, pvals, conf_left, conf_right



