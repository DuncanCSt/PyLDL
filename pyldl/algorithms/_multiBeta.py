import numpy as np

import keras
import tensorflow as tf

from pyldl.algorithms.base import BaseDeepLDL, BaseAdam
from scipy.stats import beta


EPS = np.finfo(np.float32).eps

@tf.function
def _multi_beta_log_pdf(D, a, b):
    """Per-sample sum of log Beta(a+1, b+1) densities across labels. Returns [N]."""
    a = tf.cast(a, D.dtype)
    b = tf.cast(b, D.dtype)
    alpha = a + 1.0
    beta_ = b + 1.0
    log_B = (tf.math.lgamma(alpha) + tf.math.lgamma(beta_)
             - tf.math.lgamma(alpha + beta_))
    log_pdf = ((alpha - 1.0) * tf.math.log(D + EPS)
               + (beta_ - 1.0) * tf.math.log(1.0 - D + EPS)
               - log_B)
    return tf.reduce_sum(log_pdf, axis=1)


@tf.function
def multi_beta_pdf(D, a, b):
    return tf.exp(_multi_beta_log_pdf(D, a, b))


@tf.function
def multi_beta_nll_loss(D, a, b):
    return -tf.reduce_mean(_multi_beta_log_pdf(D, a, b))


class _MultiBase(BaseAdam, BaseDeepLDL):

    _LOSS = staticmethod(multi_beta_nll_loss)

    def __init__(self, n_hidden=64, n_latent=32, **kwargs):
        super().__init__(n_hidden, n_latent, **kwargs)

    def _get_default_model(self):
        return self.get_3layer_model(
            n_features=self._n_features, n_hidden=self._n_hidden,
            n_outputs=2 * self._n_outputs,
            hidden_activation='relu', output_activation='softmax',
            dropout_rate=self._dropout_rate,
        )
    
    def _before_train(self):
        self._Dbar = tf.reduce_mean(self._D, axis=0, keepdims=True)
        self._W = 0.5

    def _loss(self, X, D, start, end):
        a, b = self._alpha(X, training=True)
        return self._LOSS(D, a, b)

    def coverage_level(self, X, Y):
        r"""Per-(sample, label) minimal credible level that covers ``Y``.

        Each label's predictive marginal is the *exact* :math:`\text{Beta}`
        distribution the model is trained under — :math:`\text{Beta}(a + 1,
        b + 1)`, with ``(a, b)`` from :meth:`_alpha` and the ``+1`` matching
        :func:`_multi_beta_log_pdf`.

        For a *central* credible interval at level :math:`c`, the true value
        :math:`y` is covered iff :math:`c \geq |2 F(y) - 1|`, where :math:`F`
        is that Beta CDF. This method returns the threshold
        :math:`c_{\min} = |2 F(y) - 1|` — the smallest confidence level whose
        100c% interval already contains ``Y``.

        For a marginally calibrated model these values are uniform on
        :math:`[0, 1]`, so the empirical coverage at confidence :math:`\gamma`
        is ``(coverage_level(X, Y) <= gamma).mean()`` and should equal
        :math:`\gamma`.

        Parameters
        ----------
        X : array-like, shape (n, n_features)
            Inputs.
        Y : array-like, shape (n, K)
            True label distributions.

        Returns
        -------
        ndarray, shape (n, K)
            Minimal central credible level covering each true label value.
        """
        a, b = self._alpha(X)
        a = np.asarray(a, dtype=float) + 1.
        b = np.asarray(b, dtype=float) + 1.
        Y = np.asarray(Y, dtype=float)
        cdf = beta.cdf(Y, a, b)
        return np.abs(2. * cdf - 1.)


@keras.saving.register_keras_serializable()
class multiBelief(_MultiBase):
    
    def _alpha(self, X, training=False):
        outputs = self._model(X, training=training)
        belief = outputs[:, :self._n_outputs]
        uncertainty = outputs[:, self._n_outputs:]
        a = self._W * ( belief + uncertainty ) / (uncertainty + EPS)
        b = self._W * (1 - belief - uncertainty) / (uncertainty + EPS)
        return a, b
    
    def predict(self, X, return_uncertainty=False):
        outputs = self._model(X, training=False)
        belief = outputs[:, :self._n_outputs]
        uncertainty = outputs[:, self._n_outputs:]

        D_pred = belief.numpy() + uncertainty.numpy()
        if return_uncertainty:
            a, b = self._alpha(X)
            a_np, b_np = a.numpy(), b.numpy()
            variance = (a_np * b_np) / ((a_np + b_np) ** 2 * (a_np + b_np + 1))
            return D_pred, variance
        return D_pred

