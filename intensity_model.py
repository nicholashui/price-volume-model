"""
Intensity-driven SV model from model.md §6–§12.

Recover λ+, λ- and P(up)/P(down) using CIR volume factor + EKF.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class VolumeSV_EKF:
    """Extended Kalman Filter for latent volume factor V_t (model.md §9)."""

    def __init__(
        self,
        kappa: float = 2.0,
        theta: float = 1.0,
        xi: float = 0.5,
        mu: float = 0.0,
        alpha: float = 1.0,
        dt: float = 1.0 / (365 * 24 * 12),  # 5-min year fraction default
        obs_noise: np.ndarray | None = None,
    ):
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.mu = mu
        self.alpha = alpha
        self.dt = dt
        self.R = obs_noise if obs_noise is not None else np.diag([1e-6, 1e-2])

    def sigma(self, V: float) -> float:
        return 0.5 * np.sqrt(max(V, 1e-8))

    def predict(self, V: float, P: float):
        V_pred = V + self.kappa * (self.theta - V) * self.dt
        F = 1.0 - self.kappa * self.dt
        Q = (self.xi**2) * max(V, 1e-8) * self.dt
        P_pred = F * P * F + Q
        return V_pred, P_pred

    def observe_func(self, V: float, imbalance: float = 0.0) -> np.ndarray:
        sig = self.sigma(V)
        r_mean = (self.mu - 0.5 * sig**2) * self.dt + self.alpha * imbalance
        dv_mean = self.kappa * (self.theta - V) * self.dt
        return np.array([r_mean, dv_mean])

    def jacobian_H(self, V: float) -> np.ndarray:
        eps = 1e-6
        h0 = self.observe_func(V)
        h1 = self.observe_func(V + eps)
        return ((h1 - h0) / eps).reshape(2, 1)

    def update(self, V_pred, P_pred, y, imbalance: float = 0.0):
        h = self.observe_func(V_pred, imbalance)
        H = self.jacobian_H(V_pred)
        S = H @ np.array([[P_pred]]) @ H.T + self.R
        try:
            K = (np.array([[P_pred]]) @ H.T) @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return V_pred, P_pred
        innov = y - h
        V_filt = V_pred + float((K @ innov)[0])
        P_filt = float(((np.eye(1) - K @ H) * P_pred)[0, 0])
        V_filt = max(V_filt, 1e-8)
        return V_filt, max(P_filt, 1e-12)

    def filter(
        self,
        returns: np.ndarray,
        volume_increments: np.ndarray,
        imbalances: np.ndarray | None = None,
    ):
        n = len(returns)
        V_filt = np.zeros(n)
        P_filt = np.zeros(n)
        V = self.theta
        P = 0.1
        for t in range(n):
            V_pred, P_pred = self.predict(V, P)
            imb = 0.0 if imbalances is None else float(imbalances[t])
            y = np.array([returns[t], volume_increments[t]])
            V, P = self.update(V_pred, P_pred, y, imb)
            V_filt[t] = V
            P_filt[t] = P
        return V_filt, P_filt


def extended_move_probabilities(
    V: float,
    dP_drift: float,
    P: float,
    dt: float,
    kappa: float,
    theta: float,
    v_bar: float,
    alpha: float,
    mu: float,
) -> tuple[float, float]:
    """model.md §12 — P(up), P(down) from intensity inversion."""
    total = kappa * (theta - V) / max(v_bar, 1e-12)
    imbalance = (dP_drift - mu * P) / max(alpha, 1e-12)
    lambda_plus = 0.5 * (total + imbalance)
    lambda_minus = 0.5 * (total - imbalance)
    lambda_plus = max(lambda_plus, 0.0)
    lambda_minus = max(lambda_minus, 0.0)
    p_up = lambda_plus * dt
    p_down = lambda_minus * dt
    s = p_up + p_down
    if s > 1.0:
        p_up /= s
        p_down /= s
    return p_up, p_down


def identify_sv_params(
    close: np.ndarray,
    volume: np.ndarray,
    signed_imbalance: np.ndarray,
    dt: float,
) -> dict:
    """
    Parameter identification (model.md §8) on a training window.
    Work in normalized units so λ± and P(up/down) are well-scaled on 5-min bars.
    """
    vol = np.asarray(volume, dtype=float)
    ret = np.diff(np.log(np.maximum(close, 1e-12)))
    ret = np.concatenate([[0.0], ret])

    # Normalize volume to mean-1 factor
    vol_n = vol / (np.mean(vol) + 1e-12)
    theta = float(np.mean(vol_n))

    # κ from lag-1 autocorr on discrete steps (not year-fraction)
    # CIR discrete: V_{t+1} ≈ V + κ(θ-V)Δ + … with Δ=1 bar
    if len(vol_n) > 20:
        c0 = float(np.corrcoef(vol_n[1:], vol_n[:-1])[0, 1])
        c0 = float(np.clip(c0, 0.01, 0.99))
        kappa = float(1.0 - c0)  # mean-reversion speed per bar in [0.01, 0.99]
        kappa = float(np.clip(kappa, 0.01, 0.8))
    else:
        kappa = 0.1

    dVn = np.diff(vol_n, prepend=vol_n[0])
    xi = float(np.std(dVn) / (np.sqrt(np.mean(vol_n)) + 1e-12))
    xi = float(np.clip(xi, 0.01, 2.0))

    # Imbalance in z-units
    imb = np.asarray(signed_imbalance, dtype=float)
    imb_z = (imb - np.mean(imb)) / (np.std(imb) + 1e-12)

    # α: dP impact of unit imbalance (log-return space)
    if len(ret) > 20 and np.std(imb_z) > 1e-12:
        alpha = float(np.cov(ret, imb_z)[0, 1] / (np.var(imb_z) + 1e-12))
    else:
        alpha = 1e-4
    alpha = float(np.sign(alpha) * max(abs(alpha), 1e-6))

    residual = ret - alpha * imb_z
    mu = float(np.mean(residual))  # per-bar drift in log space

    v_bar = 1.0  # volume already normalized

    return {
        "kappa": kappa,
        "theta": theta,
        "xi": xi,
        "mu": mu,
        "alpha": alpha,
        "v_bar": v_bar,
    }


def compute_intensity_series(
    df5: pd.DataFrame,
    params: dict,
    dt: float,
) -> pd.DataFrame:
    """
    Causal intensity features on 5-min bars (normalized discrete-time units).

    Uses model.md intensity inversion with:
      - V_t = EKF-filtered normalized volume
      - imbalance from EWMA signed-flow + short-horizon return drift
      - P(up)/P(down) proportional to λ± (renormalized to sum ≤ 1)
    """
    close = df5["close"].values.astype(float)
    volume = df5["volume"].values.astype(float)
    n = len(close)

    logp = np.log(np.maximum(close, 1e-12))
    ret = np.diff(logp, prepend=logp[0])
    # causal EWMA drift of log returns (past-only via ewm recursive)
    ewma_ret = pd.Series(ret).ewm(span=12, adjust=False).mean().shift(1).fillna(0.0).values

    if "signed_vol" in df5.columns:
        imb = df5["signed_vol"].values.astype(float)
    else:
        imb = np.sign(ret) * volume

    warm = max(100, n // 20)
    vol_scale = np.mean(volume[:warm]) + 1e-12
    imb_scale = np.std(imb[:warm]) + 1e-12
    vol_n = volume / vol_scale
    dV = np.diff(vol_n, prepend=vol_n[0])
    imb_z = imb / imb_scale
    # causal EWMA of imbalance pressure
    ewma_imb = pd.Series(imb_z).ewm(span=6, adjust=False).mean().shift(1).fillna(0.0).values

    # discrete bar dt = 1 for EKF stability on 5-min grid
    bar_dt = 1.0
    ekf = VolumeSV_EKF(
        kappa=params["kappa"],
        theta=params["theta"],
        xi=params["xi"],
        mu=params["mu"],
        alpha=params["alpha"],
        dt=bar_dt,
        obs_noise=np.diag([1e-6, 1e-2]),
    )
    V_hat, P_var = ekf.filter(ret, dV, ewma_imb)

    # Intensity inversion in normalized log-return units (model.md §6–§7)
    # total intensity from volume CIR drift; imbalance from residual drift vs flow
    total = params["kappa"] * (params["theta"] - V_hat) / max(params["v_bar"], 1e-12)
    # combine return drift signal + flow: positive → buy intensity
    flow_signal = ewma_imb * abs(params["alpha"])
    ret_signal = ewma_ret - params["mu"]
    imbalance = (ret_signal + flow_signal) / max(abs(params["alpha"]), 1e-8)

    lam_p = np.maximum(0.5 * (np.abs(total) + total + imbalance), 0.0)
    lam_m = np.maximum(0.5 * (np.abs(total) + total - imbalance), 0.0)
    # Ensure baseline intensity so probabilities not all zero
    base = 0.05 + 0.1 * np.maximum(V_hat, 0.0)
    lam_p = lam_p + base
    lam_m = lam_m + base

    # Softmax-style move probabilities for the next bar
    # P(up) ∝ λ+, P(down) ∝ λ−, with optional flat mass
    strength = lam_p + lam_m
    p_move = 1.0 - np.exp(-strength)  # more intensity → higher chance of a move
    p_up = p_move * (lam_p / (strength + 1e-12))
    p_down = p_move * (lam_m / (strength + 1e-12))

    out = pd.DataFrame(
        {
            "V_hat": V_hat,
            "V_var": P_var,
            "p_up": p_up,
            "p_down": p_down,
            "lambda_plus": lam_p,
            "lambda_minus": lam_m,
            "edge": p_up - p_down,
            "p_sum": p_up + p_down,
            "intensity_skew": (lam_p - lam_m) / (lam_p + lam_m + 1e-12),
            "flow_ewma": ewma_imb,
            "ret_ewma": ewma_ret,
        },
        index=df5.index,
    )
    return out
