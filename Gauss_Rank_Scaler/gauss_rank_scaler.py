# Input Normalization is especially important for neural networks, while Gauss Rank serves as one of the best ways to
# convert numeric variable distribution to normals.
# The main step is:
# 1. assign a spacing between -1 and 1 to the sorted features
# 2. apply the inverse of error function `erfinv` to curve the data to Gaussian-like
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import FLOAT_DTYPES, check_array, check_is_fitted
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from scipy.special import erf, erfinv

class GaussRankScaler(BaseEstimator, TransformerMixin):
    def __init__(self, epsilon=1e-4, copy=True, n_jobs=None, interp_kind='linear', interp_copy=False):
        self.epsilon = epsilon
        self.copy = copy
        self.interp_kind = interp_kind
        self.interp_copy = interp_copy
        self.n_jobs = n_jobs
        self.fill_value = 'extrapolate'

    def fit(self, X, y=None):
        '''
        Fit interpolation function to set rank for the original data which would be used for scaling
        :param X: array-like, shape = (n_examples, n_features), The columnwise data (per feature) would be used to fit interpolation
        function for later scaling
        :param y: do not use
        :return: self
        '''
        X = check_array(X, copy=self.copy, estimator=self, dtype=FLOAT_DTYPES, force_all_finite=True)
        # How could estimator=self and return self?
        self.interp_func_ = Parallel(n_jobs=self.n_jobs)(delayed(self._fit)(x) for x in X.T)
        # What does this line used for?
        return self
    def _fit(self, x):
        x = self.drop_duplicates(x)
        # Why do we need two argsort? I am guessing actually it is a mistake made.
        rank = np.argsort(x)
        # rank = np.argsort(np.argsort(x))
        bound = 1.0 - self.epsilon
        # Why do we need to scale like that? np.max(rank) should be the number of distinct value - 1, we may want to
        # transform the distribution into Norm(0,1)
        factor = np.max(rank) / 2.0 * bound
        # What does clip used for? numpy.clip(a, a_min, a_max, out=None, **kwargs)
        # Equivalent to but faster than np.minimum(a_max, np.maximum(a, a_min)).
        # Otherwise, the max_value will be scaled down to a numerical value which is slightly larger than 1.
        scaled_rank = np.clip(rank/factor-bound, -bound, bound)
        # What does interp1d use for?
        # This class returns a function whose call method uses interpolation to find the value of new points.
        # If “extrapolate”, then points outside the data range will be extrapolated.
        # If the data we used are already numpy.ndarray, then we don't need copy again.
        return interp1d(
            x, scaled_rank, kind=self.interp_kind, copy=self.copy, fill_value=self.fill_value)
    def transform(self, X, copy=None):
        # What does check_is_fitted used for?
        check_is_fitted(self, 'interp_func_')
        # I believe this option for copy is used for what we fit and transform has different data type, then we could
        # set copy or not for fit and transform seperately.
        copy = copy if copy is not None else self.copy
        # What does check_array use for?
        X = check_array(X, copy=copy, estimator=self, dtype=FLOAT_DTYPES, force_all_finite=True)
        X = np.array(Parallel(n_jobs=self.n_jobs)(delayed(self._transform)(i, x) for i, x in enumerate(X.T))).T
        return X
    def _transform(self, i, x):
        # How can we use interp_func_ since we didn't set them?
        # and what does erfinv mean?
        return erfinv(self.interp_func_[i](x))
    def inverse_transform(self, X, copy=None):
        '''
        Scale back the data to the original representation
        :param X: array-like, shape [n_samples, n-features]
            The data used to scale along the features axis
        :param copy: bool, optional (default: None)
            Copy the input X or not
        :return:
        '''
        check_is_fitted(self, 'interp_func_')
        copy = copy if copy is not None else self.copy
        X = check_array(X, copy=copy, estimator=self, dtype=FLOAT_DTYPES, force_all_finite=True)
        X = np.array(Parallel(n_jobs=self.n_jobs)(delayed(self._inverse_transform)(i,x) for i, x in enumerate(X.T))).T
        return X
    def _inverse_transform(self, i, x):
        inv_interp_func = interp1d(self.interp_func_[i].y, self.interp_func_[i].x, kind=self.interp_kind,
                                   copy=self.copy, fill_value=self.fill_value)
        return inv_interp_func(erf(x))
    @staticmethod
    def drop_duplicates(x):
        is_unique = np.zeros_like(x, dtype=bool)
        is_unique[np.unique(x, return_index=True)[1]] = True
        return x[is_unique]
