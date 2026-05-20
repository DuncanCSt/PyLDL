import numpy as np

import keras
import tensorflow as tf


from pyldl.algorithms.base import BaseDeepLDL, BaseAdam
from pyldl.algorithms._bedl import edl_loglikelihood_loss, edl_bayes_mse_loss

EPS = np.finfo(np.float32).eps

# ψ(1) = -γ (Euler–Mascheroni)
_DIGAMMA_1 = tf.constant(-0.5772156649015329, dtype=tf.float64)
_ONE_64 = tf.constant(1., dtype=tf.float64)


@tf.function
def inv_digamma(y, n_newton=5):
    y = tf.cast(y, tf.float64)
    x = tf.where(y >= -2.22,
                 tf.exp(y) + 0.5,
                 -1.0 / (y - _DIGAMMA_1))
    for _ in range(n_newton):
        x = x - (tf.math.digamma(x) - y) / tf.math.polygamma(_ONE_64, x)
    return x


@tf.function
def _solve_alpha_fp_64(log_belief, tol=1e-10, max_iter=200):
    X = tf.cast(log_belief, tf.float64)
    alpha0 = inv_digamma(X + tf.math.digamma(tf.constant(2., tf.float64)))

    def cond(i, alpha, prev_diff):
        return tf.logical_and(i < max_iter, prev_diff > tol)

    def body(i, alpha, _):
        s = tf.math.digamma(tf.reduce_sum(alpha, axis=-1, keepdims=True))
        new_alpha = inv_digamma(X + s)
        diff = tf.reduce_max(tf.abs(new_alpha - alpha))
        return i + 1, new_alpha, diff

    _, alpha, _ = tf.while_loop(
        cond, body,
        loop_vars=(tf.constant(0), alpha0, tf.constant(float("inf"), tf.float64)),
    )
    return alpha


@tf.custom_gradient
def solve_alpha_fp(log_belief):
    # Forward: Minka fixed-point on the stationarity condition
    #   ψ(α_k) − ψ(Σα) = log_belief_k
    # Backward: implicit differentiation. The Jacobian of the residual w.r.t. α
    # is J = diag(ψ'(α)) − ψ'(Σα)·11ᵀ, so ∂α/∂(log_belief) = J⁻¹. Solve J v = g
    # row-wise with Sherman–Morrison in O(K), avoiding backprop through the loop.
    in_dtype = log_belief.dtype
    alpha = _solve_alpha_fp_64(log_belief)

    def grad(d_alpha):
        g = tf.cast(d_alpha, tf.float64)
        d = tf.math.polygamma(_ONE_64, alpha)
        S = tf.reduce_sum(alpha, axis=-1, keepdims=True)
        c = tf.math.polygamma(_ONE_64, S)

        inv_d = 1.0 / d
        u = g * inv_d
        s_g = tf.reduce_sum(u, axis=-1, keepdims=True)
        s_1 = tf.reduce_sum(inv_d, axis=-1, keepdims=True)
        v = u + (c * s_g / (1.0 - c * s_1)) * inv_d
        return tf.cast(v, in_dtype)

    return tf.cast(alpha, in_dtype), grad


class _BoojumBase(BaseAdam, BaseDeepLDL):

    _LOSS = staticmethod(edl_loglikelihood_loss)

    def __init__(self, n_hidden=64, n_latent=32, **kwargs):
        super().__init__(n_hidden, n_latent, **kwargs)

    def _get_default_model(self):
        return self.get_3layer_model(
            n_features=self._n_features, n_hidden=self._n_hidden,
            n_outputs=self._n_outputs + 1,
            hidden_activation='relu', output_activation='softmax',
            dropout_rate=self._dropout_rate,
        )

    def _alpha(self, X, training=False):
        outputs = self._model(X, training=training)
        belief = tf.clip_by_value(outputs[:, :-1], EPS, 1 - EPS)
        log_belief = tf.math.log(belief)
        return solve_alpha_fp(log_belief)

    def _loss(self, X, D, start, end):
        return self._LOSS(D, self._alpha(X, training=True))

    def predict(self, X, return_uncertainty=False):
        alpha = np.asarray(self._alpha(X))
        alpha_0 = np.sum(alpha, axis=1, keepdims=True)
        D_pred = (alpha - 1) / (alpha_0 - alpha.shape[1])
        if return_uncertainty:
            variance = alpha * (alpha_0 - alpha) / (alpha_0 ** 2 * (alpha_0 + 1.))
            return D_pred, variance
        return D_pred


@keras.saving.register_keras_serializable()
class BOOJUM(_BoojumBase):
    r""":class:`BOOJUM <pyldl.algorithms.BOOJUM>` is a label-distribution adaptation of the
    Evidential Deep Learning (EDL) framework.

    """

@keras.saving.register_keras_serializable()
class BOOJUM_BAYES(BOOJUM):
    r""":class:`BOOJUM_BAYES <pyldl.algorithms.BOOJUM_BAYES>` is :class:`BOOJUM` trained with
    the Bayes-risk MSE loss instead of the Dirichlet negative log-likelihood.
    """

    _LOSS = staticmethod(edl_bayes_mse_loss)

