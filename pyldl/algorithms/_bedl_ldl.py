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


@keras.saving.register_keras_serializable()
class BEDL_LDL(BaseAdam, BaseDeepLDL):
    r""":class:`BEDL-LDL <pyldl.algorithms.BEDL_LDL>` adapts Evidential Deep Learning
    :cite:`2018:sensoy` to label distribution learning.

    The network outputs non-negative evidence :math:`\boldsymbol{e}(X) = \text{softplus}(\text{MLP}(X))`
    and Dirichlet concentrations are :math:`\boldsymbol{\alpha} = \boldsymbol{e} + 1`. Training
    minimizes the negative log-likelihood of the observed label distribution under
    :math:`\text{Dir}(\boldsymbol{\alpha}(X))`. Prediction returns the Dirichlet mean
    :math:`\boldsymbol{\alpha} / \alpha_0`, with optional per-label variance and total-evidence
    uncertainty :math:`u = K / \alpha_0`.
    """

    def __init__(self, n_hidden=64, n_latent=32, **kwargs):
        super().__init__(n_hidden, n_latent, **kwargs)

    _LOSSES = {
        'loglikelihood': edl_loglikelihood_loss,
        'bayes_mse': edl_bayes_mse_loss,
    }

    def _alpha(self, X, training=False):

        outputs = self._model(X, training=training)
        belief = outputs[:, :-1]
        uncertainty = outputs[:, -1:]

        W = tf.constant(2.0, dtype=belief.dtype)  # was: W = 2

        evidence = W * belief / (uncertainty + EPS)

        alpha = evidence + self._Dbar * W

        return alpha

    def _loss(self, X, D, start, end):
        return self._loss_fn(D, self._alpha(X, training=True))

    def _get_default_model(self):
        return self.get_3layer_model(
            n_features=self._n_features, n_hidden=self._n_hidden,
            n_outputs=self._n_outputs + 1,
            hidden_activation='relu', output_activation='softmax',
            dropout_rate=self._dropout_rate,
        )

    def _before_train(self):
        self._loss_fn = self._LOSSES[self._loss_type]
        self._Dbar = tf.reduce_mean(self._D, axis=0, keepdims=True)

    def fit(self, X, D, loss_type='loglikelihood', **kwargs):
        if loss_type not in self._LOSSES:
            raise ValueError(f"loss_type must be one of {list(self._LOSSES)}, got {loss_type!r}")
        self._loss_type = loss_type
        return super().fit(X, D, **kwargs)

    def predict(self, X, return_uncertainty=False):
        alpha = self._alpha(X).numpy()
        alpha_0 = np.sum(alpha, axis=1, keepdims=True)
        D_pred = alpha / alpha_0
        if return_uncertainty:
            variance = alpha * (alpha_0 - alpha) / (alpha_0 ** 2 * (alpha_0 + 1.))
            evidence_uncertainty = (self._n_outputs / alpha_0).reshape(-1)
            return D_pred, variance, evidence_uncertainty
        return D_pred
