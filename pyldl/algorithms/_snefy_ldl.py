import numpy as np
from scipy.stats import beta as _beta
import tensorflow as tf

from pyldl.algorithms.base import BaseDeepLDL, BaseAdam


EPS = np.finfo(np.float32).eps


class SNEFY_LDL(BaseAdam, BaseDeepLDL):
    """:class:`SNEFY-LDL <pyldl.algorithms.SNEFY_LDL>` is proposed in paper :cite:`2025:zhang`. 
    SNEFY refers to *squared neural family*.
    """

    def __init__(self, n_hidden=64, n_latent=32, **kwargs):
        super().__init__(n_hidden, n_latent, **kwargs)

    @tf.function
    def _kernel(self, features):
        gamma = lambda x: tf.math.exp(tf.math.lgamma(x))
        temp = self._b + features
        F = tf.expand_dims(temp, axis=1) + tf.expand_dims(temp, axis=2)
        W = tf.expand_dims(self._W, axis=1) + tf.expand_dims(self._W, axis=2)
        numerator = tf.reduce_prod(gamma(W + 1), axis=0)
        denominator = gamma(tf.cast(self._n_outputs, tf.float32) + tf.reduce_sum(W, axis=0))
        return tf.reshape(tf.math.exp(F) * (numerator / denominator), (-1, self._n_hidden, self._n_hidden))

    @tf.function
    def _calculate_VKV(self, features):
        K = self._kernel(features)
        VTV = tf.transpose(self._V) @ self._V
        return K * VTV

    @tf.function
    def _loss(self, X, D, start, end):
        features = self._encoder(X, training=True)
        latent = tf.math.exp(self._log_D[start:end] @ self._W + features + self._b)
        net = tf.reshape((tf.norm(latent @ tf.transpose(self._V), axis=1)**2), (-1, ))
        log = tf.math.log(net + EPS)
        VKV = self._calculate_VKV(features)
        return - tf.reduce_mean(log - tf.math.log(tf.reduce_sum(VKV, axis=(1, 2)) + EPS))

    def _before_train(self):
        self._log_D = tf.math.log(self._D)
        self._encoder = self.get_3layer_model(self._n_features, self._n_hidden, self._n_hidden,
                                              hidden_activation='relu', output_activation=None,
                                              dropout_rate=self._dropout_rate)
        self._W = tf.Variable(tf.random.normal((self._n_outputs, self._n_hidden)), trainable=True)
        self._V = tf.Variable(tf.random.normal((self._n_latent, self._n_hidden), 0., 1.) *\
                              tf.sqrt(1. / (self._n_latent * self._n_hidden)), trainable=True)
        self._b = tf.Variable(tf.zeros((1, self._n_hidden)), trainable=True)

    def train_step(self, batch, loss, trainable_variables, optimizer, epoch, epochs, start, end):
        super().train_step(batch, loss, trainable_variables, optimizer, epoch, epochs, start, end)
        self._W.assign(tf.maximum(self._W, -.495))

    def fit(self, X, Y, *, batch_size=64, **kwargs):
        return super().fit(X, Y, batch_size=batch_size, **kwargs)

    def predict(self, X, return_uncertainty=False):
        features = self._encoder(X)
        VKV = self._calculate_VKV(features)
        W = tf.expand_dims(self._W, axis=1) + tf.expand_dims(self._W, axis=2)
        alpha = W + 1
        alpha_0 = self._n_outputs + tf.reduce_sum(W, axis=0)
        E1 = alpha / alpha_0
        E2 = (alpha * (alpha + 1)) / (alpha_0 * (alpha_0 + 1))
        tempE1 = tf.einsum('bmn,dmn->bd', VKV, E1)
        tempE2 = tf.einsum('bmn,dmn->bd', VKV, E2)
        VKV_sum = tf.reshape(tf.reduce_sum(VKV, axis=(1, 2)), (-1, 1))
        D_pred = (tempE1 / VKV_sum).numpy()
        if return_uncertainty:
            uncertainty = (tempE2 / VKV_sum).numpy() - D_pred**2
            return D_pred, uncertainty
        return D_pred

    def coverage_level(self, X, Y):
        r"""Per-(sample, label) minimal credible level that covers ``Y``.

        Unlike the EDL/BOOJUM models, SNEFY-LDL's predictive distribution is a
        squared neural family, whose per-label marginal has no closed-form CDF.
        We therefore approximate each label's marginal by a :math:`\text{Beta}`
        distribution moment-matched to the predicted mean and variance returned
        by :meth:`predict` (the same mean/variance the conformal pipeline
        already consumes for SNEFY).

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
        mean, var = self.predict(X, return_uncertainty=True)
        mean = np.clip(np.asarray(mean, dtype=float), EPS, 1. - EPS)
        # A Beta on [0, 1] requires var < mean*(1-mean); clip for safety. A
        # multiplicative upper bound keeps var strictly inside (0, max_var)
        # even when mean is near 0/1 (where max_var - EPS could go negative).
        max_var = mean * (1. - mean)
        var = np.clip(np.asarray(var, dtype=float), EPS, max_var * (1. - 1e-3))
        common = max_var / var - 1.
        a = mean * common
        b = (1. - mean) * common
        Y = np.asarray(Y, dtype=float)
        cdf = _beta.cdf(Y, a, b)
        return np.abs(2. * cdf - 1.)
