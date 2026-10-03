"""Exact Gaussian Process Regression (numpy + scipy.optimize).

Kernel: sigma_f^2 * ARD squared-exponential + sigma_n^2 * white noise.
Hyperparameters maximise the log marginal likelihood (L-BFGS-B with analytic
gradients, several restarts). Predictions return a mean and a standard
deviation, so every estimate ships with a calibrated interval.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize

_LOG_2PI = np.log(2 * np.pi)


class GaussianProcess:
    def __init__(self, n_restarts: int = 3, seed: int = 0, length_bounds=(0.05, 50.0), noise_bounds=(1e-3, 1.0)):
        self.n_restarts = n_restarts
        self.rng = np.random.default_rng(seed)
        self.length_bounds = length_bounds
        self.noise_bounds = noise_bounds
        self.theta_: np.ndarray | None = None

    # --------------------------------------------------------------- internals
    @staticmethod
    def _sqdist(a: np.ndarray, b: np.ndarray, ls: np.ndarray) -> np.ndarray:
        a, b = a / ls, b / ls
        d = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T
        return np.maximum(d, 0.0)

    def _unpack(self, theta):
        d = self.X_.shape[1]
        return np.exp(theta[:d]), np.exp(theta[d]), np.exp(theta[d + 1])

    def _nll(self, theta):
        ls, sf, sn = self._unpack(theta)
        X, y = self.X_, self.y_
        n = len(y)
        kf = sf**2 * np.exp(-0.5 * self._sqdist(X, X, ls))
        K = kf + (sn**2 + 1e-8) * np.eye(n)
        try:
            c = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e10, np.zeros_like(theta)
        alpha = cho_solve(c, y)
        nll = 0.5 * y @ alpha + np.log(np.diag(c[0])).sum() + 0.5 * n * _LOG_2PI
        W = np.outer(alpha, alpha) - cho_solve(c, np.eye(n))  # (aa^T - K^-1)
        grad = np.empty_like(theta)
        for k in range(len(ls)):
            dk = kf * (X[:, k : k + 1] - X[:, k : k + 1].T) ** 2 / ls[k] ** 2
            grad[k] = -0.5 * np.sum(W * dk)
        grad[-2] = -0.5 * np.sum(W * 2 * kf)
        grad[-1] = -0.5 * np.trace(W) * 2 * sn**2
        return nll, grad

    # ------------------------------------------------------------------- API
    def fit(self, X: np.ndarray, y: np.ndarray) -> GaussianProcess:
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.x_mean_, self.x_std_ = X.mean(0), X.std(0) + 1e-12
        self.y_mean_, self.y_std_ = y.mean(), y.std() + 1e-12
        self.X_ = (X - self.x_mean_) / self.x_std_
        self.y_ = (y - self.y_mean_) / self.y_std_
        d = X.shape[1]
        bounds = [tuple(np.log(self.length_bounds))] * d + [
            (np.log(0.05), np.log(20.0)),
            tuple(np.log(self.noise_bounds)),
        ]
        starts = [np.r_[np.zeros(d) + np.log(1.5), 0.0, np.log(0.1)]]
        for _ in range(self.n_restarts - 1):
            starts.append(
                np.r_[
                    self.rng.uniform(-0.5, 2.0, d),
                    self.rng.uniform(-0.5, 0.5),
                    self.rng.uniform(np.log(0.02), np.log(0.3)),
                ]
            )
        best = None
        for s in starts:
            r = minimize(self._nll, s, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 200})
            if best is None or r.fun < best.fun:
                best = r
        self.theta_ = best.x
        self.nll_ = float(best.fun)
        self._precompute()
        return self

    def _precompute(self):
        ls, sf, sn = self._unpack(self.theta_)
        K = sf**2 * np.exp(-0.5 * self._sqdist(self.X_, self.X_, ls)) + (sn**2 + 1e-8) * np.eye(len(self.y_))
        self.L_ = np.linalg.cholesky(K)
        self.alpha_ = cho_solve((self.L_, True), self.y_)

    def predict(self, X: np.ndarray, include_noise: bool = True):
        """Returns (mean, std) in the original target units."""
        Xs = (np.atleast_2d(np.asarray(X, float)) - self.x_mean_) / self.x_std_
        ls, sf, sn = self._unpack(self.theta_)
        ks = sf**2 * np.exp(-0.5 * self._sqdist(Xs, self.X_, ls))
        mean = ks @ self.alpha_
        v = solve_triangular(self.L_, ks.T, lower=True)
        var = np.maximum(sf**2 - (v * v).sum(0), 1e-12)
        if include_noise:
            var = var + sn**2
        return mean * self.y_std_ + self.y_mean_, np.sqrt(var) * self.y_std_

    def length_scales(self) -> np.ndarray:
        """ARD length scales in standardised units (small = feature matters)."""
        return self._unpack(self.theta_)[0]

    # ------------------------------------------------------------ persistence
    def to_arrays(self, prefix: str) -> dict:
        return {
            f"{prefix}theta": self.theta_,
            f"{prefix}X": self.X_,
            f"{prefix}y": self.y_,
            f"{prefix}xm": self.x_mean_,
            f"{prefix}xs": self.x_std_,
            f"{prefix}ym": np.array(self.y_mean_),
            f"{prefix}ys": np.array(self.y_std_),
        }

    @classmethod
    def from_arrays(cls, z, prefix: str) -> GaussianProcess:
        gp = cls()
        gp.theta_, gp.X_, gp.y_ = z[f"{prefix}theta"], z[f"{prefix}X"], z[f"{prefix}y"]
        gp.x_mean_, gp.x_std_ = z[f"{prefix}xm"], z[f"{prefix}xs"]
        gp.y_mean_, gp.y_std_ = float(z[f"{prefix}ym"]), float(z[f"{prefix}ys"])
        gp._precompute()
        return gp
