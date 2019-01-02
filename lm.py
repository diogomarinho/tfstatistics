#!/usr/bin/python3
import pandas as pd
import numpy as np
import tensorflow as tf
from leastsquares import leastSquaresSolver, byInverse

# response variabl (Y): SalePrice
# predictors: LotArea (Numerical) LotFrontage (Numerical) Street(Categorical)
# example data to develop the linear regression method
def getXYData(fin='~/tfstatistics/house_pricing_train.csv'):
    df = pd.read_csv(fin)
    xydata = df['SalePrice LotArea LotFrontage Street'.split()]
    xydata = xydata.dropna(axis=0)
    xydata.Street = xydata.Street.factorize()[0]
    return xydata

class lm:
    def __init__(self, data, response_id):
        self.y = data[[response_id]].values
        self.x = data.drop([response_id], axis=1).values
        # adding ones for the intercept
    def fit(self, lss):
        lss.solve(self.x, self.y)
        return 0

df = getXYData()
lreg = lm(df, 'SalePrice')
lreg.fit(byInverse())

