# Authors: Jonathan Chung <jono.chung@gmail.com>
#
# License: BSD 3 clause

"""Implementation of minimum Redundancy and Maximum Relevance (mRMR) for feature selection"""

import numpy as np
from ..utils import check_X_y, safe_sqr
from ..utils.metaestimators import if_delegate_has_method
from ..base import BaseEstimator
from ..base import MetaEstimatorMixin
from ..base import clone
from .base import SelectorMixin
from sklearn.metrics import mutual_info_score
from scipy.stats import f_oneway
from scipy.stats import pearsonr


class MRMR(BaseEstimator, MetaEstimatorMixin, SelectorMixin):
    """Feature ranking with mRMR.


    Parameters
    ----------


    Attributes
    ----------


    Examples
    --------


    References
    ----------

    .. [1] Peng, H.C., Long, F., & Ding, C., "Feature selection based on mutual information:
           criteria of max-dependency, max-relevance, and min-redundancy.",
           IEEE Transactions on Pattern Analysis and Machine Intelligence,
           27(8), 1226--1238, 2005.
    """
    def __init__(self, n_features_to_select, kmax=1000, algorithm="MID"):
        if n_features_to_select <= 0:
            raise "n_features_to_select must be greater than 0"
        self.n_features_to_select = n_features_to_select
        self.kmax = kmax
        self.algorithm = algorithm  
        # Initialisations
        def f_oneway_func(y, X):
            # f_oneway takes (*args) where *args are samples
            samplesX = []
            for i in np.unique(y):
                samplesX.append(X[y==i])
            return -f_oneway(*samplesX)[1]
        
        self.relevance_algo = {"MID": mutual_info_score,
                                "MIQ": mutual_info_score,
                                "FCD": f_oneway_func,
                                "FCQ": f_oneway_func}
   
        self.redundancy_algo = {"MID": mutual_info_score,
                                 "MIQ": mutual_info_score,
                                 "FCD": lambda x, y: -pearsonr(x, y)[1],
                                 "FCQ": lambda x, y: -pearsonr(x, y)[1]}

        additive_comb_func = lambda x, y: -(x + y)
        multiplicative_comb_func = lambda x, y: x / y
        self.combine_algo = {"MID": additive_comb_func,
                   "MIQ": multiplicative_comb_func,
                   "FCD": additive_comb_func,
                   "FCQ": multiplicative_comb_func}

    def fit(self, X, y):
        """Fit the RFE model and then the underlying estimator on the selected
           features.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.
        """     
        X, y = check_X_y(X, y, "csc")
        # Initialisations
        supported_algorithms = self.relevance_algo.keys()
        if self.algorithm not in supported_algorithms:
            raise ValueError("Algorithm {0} is not supported".format(self.algorithm))
        
        discrete_algorithms = ["MID", "MIQ"]
        if self.algorithm in discrete_algorithms:
            if X.dtype == np.float: # Check if values are discrete
                raise ValueError("Please use algorithms {0} for float (continuous) data".format(
                    filter(lambda x: x not in discrete_algorithms, supported_algorithms)))

        N = X.shape[1]
        K = self.n_features_to_select
        self.ranking_  = self._run_mRMR(N, K, y, X)

        return self

    def _run_mRMR(self, N, K, y, X):
        t = np.zeros(N)
        for i in range(N):         
            t[i] = self.relevance_algo[self.algorithm](y, X[:, i])
        idxs = np.argsort(t)[::-1]
        fea = np.zeros(K, dtype=np.int)    
        fea[0] = idxs[0]
        kmax = min(self.kmax, N)

        idxleft = idxs[1:kmax]
        for k in range(1, K):
            prev_fea = fea[k - 1] # The previously chosen feature
            n = idxleft.shape[0]
            t_mi = np.zeros(n)
            c_mi = np.zeros(n)
            for idx, i in enumerate(idxleft):
                t_mi[idx] = self.relevance_algo[self.algorithm](y, X[:, i]) 
                da = X[:, prev_fea]
                dt = X[:, i]
                c_mi[idx] = self.redundancy_algo[self.algorithm](da, dt)

            mRMR_idx = np.argmax(self.combine_algo[self.algorithm](t_mi, c_mi))
            fea[k] = idxleft[mRMR_idx]
            idxleft = np.delete(idxleft, mRMR_idx)
         
        return fea

    @if_delegate_has_method(delegate='estimator')
    def predict(self, X):
        """Reduce X to the selected features and then predict using the
           underlying estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape [n_samples]
            The predicted target values.
        """
        return self.estimator_.predict(self.transform(X))

    @if_delegate_has_method(delegate='estimator')
    def score(self, X, y):
        """Reduce X to the selected features and then return the score of the
           underlying estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        y : array of shape [n_samples]
            The target values.
        """
        return self.estimator_.score(self.transform(X), y)

    def _get_support_mask(self):
        return self.support_

    @if_delegate_has_method(delegate='estimator')
    def decision_function(self, X):
        return self.estimator_.decision_function(self.transform(X))

    @if_delegate_has_method(delegate='estimator')
    def predict_proba(self, X):
        return self.estimator_.predict_proba(self.transform(X))

    @if_delegate_has_method(delegate='estimator')
    def predict_log_proba(self, X):
        return self.estimator_.predict_log_proba(self.transform(X))
