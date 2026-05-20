import numpy as np

import keras
import tensorflow as tf

from pyldl.algorithms.base import BaseDeepLDL, BaseAdam


EPS = np.finfo(np.float32).eps


@tf.function
def edl_loglikelihood_loss(D, alpha):
    alpha_0 = tf.reduce_sum(alpha, axis=1, keepdims=True)
    log_B = tf.reduce_sum(tf.math.lgamma(alpha), axis=1, keepdims=True) - tf.math.lgamma(alpha_0)
    cross = tf.reduce_sum((alpha - 1.) * tf.math.log(D + EPS), axis=1, keepdims=True)
    return tf.reduce_mean(log_B - cross)


@tf.function
def edl_bayes_mse_loss(D, alpha):
    alpha_0 = tf.reduce_sum(alpha, axis=1, keepdims=True)
    mean = alpha / alpha_0
    mse = tf.reduce_sum(tf.square(D - mean), axis=1)
    var = tf.reduce_sum(alpha * (alpha_0 - alpha) / (alpha_0 ** 2 * (alpha_0 + 1.)), axis=1)
    return tf.reduce_mean(mse + var)


class _EDLBase(BaseAdam, BaseDeepLDL):

    _LOSS = staticmethod(edl_loglikelihood_loss)

    def __init__(self, n_hidden=64, n_latent=32, **kwargs):
        super().__init__(n_hidden, n_latent, **kwargs)

    def _loss(self, X, D, start, end):
        return self._LOSS(D, self._alpha(X, training=True))

    def predict(self, X, return_uncertainty=False):
        alpha = self._alpha(X).numpy()
        alpha_0 = np.sum(alpha, axis=1, keepdims=True)
        D_pred = (alpha - 1) / (alpha_0 - alpha.shape[1])
        if return_uncertainty:
            variance = alpha * (alpha_0 - alpha) / (alpha_0 ** 2 * (alpha_0 + 1.))
            uncertainty = (self._n_outputs / alpha_0).reshape(-1)
            return D_pred, variance, uncertainty
        return D_pred


@keras.saving.register_keras_serializable()
class EDL(_EDLBase):
    r""":class:`EDL <pyldl.algorithms.EDL>` adapts Evidential Deep Learning
    :cite:`2018:sensoy` to label distribution learning.

    The network outputs non-negative evidence :math:`\boldsymbol{e}(X) = \text{softplus}(\text{MLP}(X))`
    and Dirichlet concentrations are :math:`\boldsymbol{\alpha} = \boldsymbol{e} + 1`. Training
    minimizes the negative log-likelihood of the observed label distribution under
    :math:`\text{Dir}(\boldsymbol{\alpha}(X))`. Prediction returns the Dirichlet mean
    :math:`\boldsymbol{\alpha} / \alpha_0`, with optional per-label variance and total-evidence
    uncertainty :math:`u = K / \alpha_0`.
    """

    def _alpha(self, X, training=False):
        return self._model(X, training=training) + 1.

    def _get_default_model(self):
        return self.get_3layer_model(
            n_features=self._n_features, n_hidden=self._n_hidden,
            n_outputs=self._n_outputs,
            hidden_activation='relu', output_activation='softplus',
            dropout_rate=self._dropout_rate,
        )


@keras.saving.register_keras_serializable()
class EDL_BAYES(EDL):
    r""":class:`EDL_BAYES <pyldl.algorithms.EDL_BAYES>` is :class:`EDL` trained with
    the Bayes-risk MSE loss instead of the Dirichlet negative log-likelihood.
    """

    _LOSS = staticmethod(edl_bayes_mse_loss)


@keras.saving.register_keras_serializable()
class BEDL(_EDLBase):
    r""":class:`BEDL <pyldl.algorithms.BEDL>` is a beta-style Evidential Deep Learning
    variant. The network outputs a softmax over :math:`K + 1` channels — :math:`K`
    belief masses and one uncertainty mass. Dirichlet concentrations are formed from a
    learned per-label scale :math:`\boldsymbol{W}` (computed from the training-label
    statistics) as :math:`\boldsymbol{\alpha} = \boldsymbol{W}\boldsymbol{b}/u +
    \boldsymbol{W}\bar{\boldsymbol{D}}`.
    """

    def _alpha(self, X, training=False):
        outputs = self._model(X, training=training)
        belief = outputs[:, :-1]
        uncertainty = outputs[:, -1:]
        evidence = self._W * belief / (uncertainty + EPS)
        return evidence + self._Dbar * self._W

    def _get_default_model(self):
        return self.get_3layer_model(
            n_features=self._n_features, n_hidden=self._n_hidden,
            n_outputs=self._n_outputs + 1,
            hidden_activation='relu', output_activation='softmax',
            dropout_rate=self._dropout_rate,
        )

    def _before_train(self):
        self._Dbar = tf.reduce_mean(self._D, axis=0, keepdims=True)
        self._W = 2


@keras.saving.register_keras_serializable()
class BEDL_BAYES(BEDL):
    r""":class:`BEDL_BAYES <pyldl.algorithms.BEDL_BAYES>` is :class:`BEDL` trained with
    the Bayes-risk MSE loss instead of the Dirichlet negative log-likelihood.
    """

    _LOSS = staticmethod(edl_bayes_mse_loss)
