import numpy as np
from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow.python.training import optimizer

# class TFAssociationTestOpt: in high dimension data it's very common
# to treat the features independently. This means that the number of regression
# models performed  are equal to the number of features: this happens in most
# GWAS and EWAS study where the number of regressions modles goes to 1 000 000
# to 11 000 000, which still a small fraction of the entire human genome.
# This tests can be paralellized to obtain faster computation. tensorflow
# already provide all matrix operations necessary to implement inverse matrix
# methods for estimating standard errors and coefficients and, more important,
# has all GPU interface embeeded in its code, allowing potentially very fast,
# calculations.

class TFAssociationTestOptimizer(optimizer.Optimizer):
    def __init__(self, X, Y, use_locking=False, name="PowerSign"):
        super(TFAssociationTestOptimizer, use_locking, name)

# . . . . . . . . . . .  . . . . . .
# class to solve the least squares using tensorflow api
class TFLeastSquaresSolver(ABC):
    def get_as_tensors(self, X, Y):
        tf_x = tf.placeholder(dtype=tf.float32, shape=X.shape)
        tf_y = tf.placeholder(dtype=tf.float32, shape=Y.shape)
        return tf_x, tf_y
    @abstractmethod
    def solve(self, X, Y):
        pass

class TFNormalEquations(TFLeastSquaresSolver):
    def solve(self, X, Y):
        x, y = self.get_as_tensors(X, Y)
        x_t = tf.transpose(x)
        covar_parameters = tf.matrix_inverse(tf.matmul(tf.transpose(x), x))
        theta = tf.matmul(covar_parameters, tf.matmul(tf.transpose(x), y))
        with tf.Session() as sess:
            results = sess.run([covar_parameters, theta], feed_dict={x:X, y:Y})
        # import pdb; pdb.set_trace()
        return results

# .
class leastSquaresSolver(ABC):
    # returns the coefficients and the product (X'X)^1
    @abstractmethod
    def solve(self, X, Y):
        print('Solving least squares')
        return None, None
# .
class byNormalEquation(leastSquaresSolver):
    # this has the same effect of :
    # (X_T . X )' * (X_T . Y)
    def solve(self, X, Y):
        x_t = X.T
        xprod = x_t.dot(X)
        l = np.linalg.cholesky(xprod)
        d = x_t.dot(Y)
        # print(d.shape)
        # solve L . z = d  ---->  z = L' * d
        z = np.linalg.inv(l).dot(d)
        # solve L^t . x = z --->  x =  L^t' * z
        return (np.linalg.inv(l.T).dot(z), np.linalg.inv(l.dot(l.T)))


# . columns of X must be linearly independent
class byQRFactorization(leastSquaresSolver):
    def solve(self, X, Y):
        # QR factorization
        q, r = np.linalg.qr(X)
        # from stats model
        return (np.linalg.solve(r, q.T.dot(Y)), np.linalg.inv(np.dot(r.T, r)))
        # my old version
        # return np.linalg.inv(r).dot(q.T.dot(Y)),np.linalg.inv(np.dot(r.T,r))

# . SVD
class bySVD(leastSquaresSolver):
    def solve(self, X, Y):
        rcond = np.asarray(1e-15)
        u, s, vt = np.linalg.svd(X, full_matrices=False)
        # discard small singular values
        cutoff = rcond[..., np.newaxis] * np.amax(s, axis=-1, keepdims=True)
        large = s > cutoff
        s = np.divide(1, s, where=large, out=s)
        s[~large] = 0
        # pseudo-inverse
        res = np.dot(np.transpose(vt), np.multiply(s[:, np.core.newaxis], np.transpose(u)))
        # res = np.matmul(np.transpose(vt), np.multiply(s[..., np.newaxis], np.transpose(u)))
        return res.dot(Y),  np.dot(res, np.transpose(res))
#
