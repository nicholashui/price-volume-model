# Price-Volume Stochastic Model

**Documentation of the full model development from classical Black–Scholes to intensity-driven stochastic-volatility dynamics with Kalman filtering.**

This document records the complete derivation path discussed in the conversation, including all intermediate equations, intensity representations, parameter identification, and the Extended Kalman Filter implementation.

---

## 1. Starting Point: Classical Black–Scholes

The classical Black–Scholes model describes the asset price $P_t$ by geometric Brownian motion:

$$
dP_t = \mu P_t\, dt + \sigma P_t\, dW_t
$$

Volume is absent. Temporary price fluctuations are captured only through the continuous diffusion term.

---

## 2. Joint Dynamics of the $(P, V)$ Pair

To incorporate volume $V_t$ we move to a bivariate description.

### 2.1 Continuous diffusion approximation

$$
\begin{cases}
dP_t = \mu(P_t, V_t)\, dt + \sigma(P_t, V_t)\, dW_t^P \\
dV_t = \alpha(P_t, V_t)\, dt + \beta(P_t, V_t)\, dW_t^V
\end{cases}
$$

with instantaneous correlation $d\langle W^P, W^V\rangle_t = \rho\, dt$.

### 2.2 Mixture of Distributions Hypothesis (MDH)

Both price changes and volume are driven by a common latent information-arrival process $I_t$. In continuous time this yields a subordinated Brownian motion whose directing process is proportional to cumulative volume.

### 2.3 Marked point-process description (high-frequency foundation)

Trade arrival times form a point process. Each trade carries a mark (size and sign). The cumulative processes are:

$$
V_t = \sum_{t_i\le t} v_i, \qquad
P_t = P_0 + \sum_{t_i\le t}\Bigl(\text{impact}(v_i,\text{sign}_i) + \text{noise}_i\Bigr)
$$

Temporary impact is a function of the instantaneous trading rate and subsequently decays; permanent impact remains.

---

## 3. Buy and Sell Intensities

Define:
- $\lambda^+_t$ — intensity of buy-initiated trades
- $\lambda^-_t$ — intensity of sell-initiated trades

**Volume compensator**

$$
dV_t = \bar{v}\,(\lambda^+_t + \lambda^-_t)\, dt
$$

**Price dynamics with order-flow imbalance**

$$
dP_t = \alpha(\lambda^+_t - \lambda^-_t)\, dt + \sigma_t\, dW_t
$$

(or with an additional temporary-impact term $\beta(v_t)\, dt$).

---

## 4. Stochastic-Volatility Black–Scholes (Volume-Driven)

The continuous limit that retains a volume channel is the stochastic-volatility model

$$
\begin{aligned}
dP_t &= \mu P_t\, dt + \sigma(V_t)\, P_t\, dW_t, \\
dV_t &= \kappa(\theta - V_t)\, dt + \xi\sqrt{V_t}\, dZ_t,
\end{aligned}
$$

with

$$
d\langle W,Z\rangle_t = \rho\, dt.
$$

When $\sigma(V)$ is constant the system collapses to classical Black–Scholes.

---

## 5. Intensity-Substituted SV-Black–Scholes

Substituting the intensity representation into the SV model yields

$$
\begin{aligned}
dP_t &= \Bigl(\mu P_t + \alpha(\lambda^+_t - \lambda^-_t)\Bigr)\, dt
+ \sigma(\lambda^+_t + \lambda^-_t)\, P_t\, dW_t, \\
dV_t &= \bar{v}\,(\lambda^+_t + \lambda^-_t)\, dt.
\end{aligned}
$$

(The intensities $\lambda^\pm$ may themselves follow mean-reverting or Hawkes dynamics.)

---

## 6. Solving for Buy and Sell Intensities

Inverting the drift relations:

**Total intensity (from volume)**

$$
\lambda^+_t + \lambda^-_t = \frac{1}{\bar{v}}\frac{dV_t}{dt}
$$

**Imbalance (from price drift)**

$$
\lambda^+_t - \lambda^-_t = \frac{1}{\alpha}\Bigl(\tfrac{dP_t}{dt}\big|_{\rm drift} - \mu P_t\Bigr)
$$

**Explicit solution**

$$
\begin{aligned}
\lambda^+_t &= \frac12\Biggl(
\frac{1}{\bar{v}}\frac{dV_t}{dt}
+ \frac{1}{\alpha}\Bigl(\tfrac{dP_t}{dt}\big|_{\rm drift} - \mu P_t\Bigr)
\Biggr), \\
\lambda^-_t &= \frac12\Biggl(
\frac{1}{\bar{v}}\frac{dV_t}{dt}
- \frac{1}{\alpha}\Bigl(\tfrac{dP_t}{dt}\big|_{\rm drift} - \mu P_t\Bigr)
\Biggr).
\end{aligned}
$$

---

## 7. Extension Using Full SV Parameters

Matching the volume drift to the CIR mean-reversion term gives the intensity expressions that are consistent with every SV parameter:

$$
\begin{aligned}
\lambda^+_t &= \frac12\left(
\frac{\kappa(\theta - V_t)}{\bar{v}}
+ \frac{1}{\alpha}\Bigl(\tfrac{dP_t}{dt}\big|_{\rm drift} - \mu P_t\Bigr)
\right), \\
\lambda^-_t &= \frac12\left(
\frac{\kappa(\theta - V_t)}{\bar{v}}
- \frac{1}{\alpha}\Bigl(\tfrac{dP_t}{dt}\big|_{\rm drift} - \mu P_t\Bigr)
\right).
\end{aligned}
$$

---

## 8. Identification of Stochastic-Volatility Parameters

| Parameter | Identification |
|-----------|----------------|
| $\theta$ | Long-run mean of the volume factor $V$ |
| $\kappa$ | $-\frac1\Delta\log\text{Corr}(V_{t+\Delta},V_t)$ |
| $\xi$ | $\sqrt{\sum(\Delta V)^2\big/\sum V\cdot\Delta t}$ |
| $\mu$ | Residual average drift of price after removing imbalance |
| $\alpha$ | Regression coefficient of returns on $(\lambda^+-\lambda^-)$ |
| $\sigma(V)$ | Fitted function of realised volatility versus $V$ |
| $\rho$ | Instantaneous correlation $\text{Corr}(dP,dV)$ |

The same $\kappa,\theta$ can be recovered by regressing observed total intensity on the current level of $V$.

---

## 9. Extended Kalman Filter for Latent Volume Factor

Because the CIR process is non-linear, an Extended Kalman Filter (EKF) is used to filter the latent volume factor $V_t$ from observed returns and volume increments.

### State transition (Euler)

$$
V_{t+1} = V_t + \kappa(\theta - V_t)\Delta t + \xi\sqrt{\max(V_t,\varepsilon)}\sqrt{\Delta t}\,\eta_t
$$

### Observation equations

$$
\begin{aligned}
r_t &= \bigl(\mu - \tfrac12\sigma(V_t)^2\bigr)\Delta t + \alpha\cdot\text{imbalance}_t + \sigma(V_t)\sqrt{\Delta t}\,\varepsilon_t^{(1)}, \\
\Delta v_t &= \kappa(\theta - V_t)\Delta t + \text{noise}.
\end{aligned}
$$

### EKF recursion (summary)

**Predict**

$$
\begin{aligned}
\hat V_{t|t-1} &= \hat V_{t-1|t-1} + \kappa(\theta - \hat V_{t-1|t-1})\Delta t, \\
P_{t|t-1} &= F_t P_{t-1|t-1} F_t^\top + Q_t,
\end{aligned}
$$

where $F_t = 1-\kappa\Delta t$ and $Q_t = \xi^2\hat V_{t|t-1}\Delta t$.

**Update**

$$
\begin{aligned}
K_t &= P_{t|t-1} H_t^\top(H_t P_{t|t-1} H_t^\top + R)^{-1}, \\
\hat V_{t|t} &= \hat V_{t|t-1} + K_t\bigl(y_t - h(\hat V_{t|t-1})\bigr), \\
P_{t|t} &= (I - K_t H_t)P_{t|t-1}.
\end{aligned}
$$

### Reference Python implementation

```python
import numpy as np

class VolumeSV_EKF:
    def __init__(self, kappa, theta, xi, mu, alpha, sigma_func,
                 dt=1/252, obs_noise=None):
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.mu = mu
        self.alpha = alpha
        self.sigma = sigma_func
        self.dt = dt
        self.R = obs_noise if obs_noise is not None else np.diag([1e-4, 1e-3])

    def predict(self, V, P):
        V_pred = V + self.kappa * (self.theta - V) * self.dt
        F = 1.0 - self.kappa * self.dt
        Q = (self.xi ** 2) * max(V, 1e-8) * self.dt
        P_pred = F * P * F + Q
        return V_pred, P_pred

    def observe_func(self, V, imbalance=0.0):
        sig = self.sigma(V)
        r_mean = (self.mu - 0.5 * sig**2) * self.dt + self.alpha * imbalance
        dv_mean = self.kappa * (self.theta - V) * self.dt
        return np.array([r_mean, dv_mean])

    def jacobian_H(self, V):
        eps = 1e-6
        h0 = self.observe_func(V)
        h1 = self.observe_func(V + eps)
        return ((h1 - h0) / eps).reshape(2, 1)

    def update(self, V_pred, P_pred, y, imbalance=0.0):
        h = self.observe_func(V_pred, imbalance)
        H = self.jacobian_H(V_pred)
        S = H @ P_pred @ H.T + self.R
        K = (P_pred @ H.T) @ np.linalg.inv(S)
        V_filt = V_pred + (K @ (y - h))[0]
        P_filt = (np.eye(1) - K @ H) @ P_pred
        return V_filt, P_filt[0, 0]

    def filter(self, returns, volume_increments, imbalances=None):
        n = len(returns)
        V_filt = np.zeros(n)
        P_filt = np.zeros(n)
        V = self.theta
        P = 0.1
        for t in range(n):
            V_pred, P_pred = self.predict(V, P)
            imb = 0.0 if imbalances is None else imbalances[t]
            y = np.array([returns[t], volume_increments[t]])
            V, P = self.update(V_pred, P_pred, y, imb)
            V_filt[t] = V
            P_filt[t] = P
        return V_filt, P_filt
```

The filtered series $\{\hat V_t\}$ can be used inside an outer optimisation loop (prediction-error likelihood) to estimate the full set of SV parameters.

---

## 10. Summary of the Model Hierarchy

| Level | Description | Recovers classical BS when $\ldots$ |
|-------|-------------|----------------------------------|
| Marked point process | Tick-level buy/sell intensities + marks | High intensity, balanced flow, temporary impact averaged out |
| Intensity SDEs | $\lambda^\pm$ dynamics | Imbalance $\to 0$, $\sigma$ constant |
| Volume-driven SV-BS | $V_t$ as CIR factor driving $\sigma(V)$ | $\sigma(V)=\text{const}$ |
| Classical Black–Scholes | Pure GBM | All volume / intensity structure removed |

---

## 11. Conversation Chronology (for reference)

1. Temporary asset price change linked to Black–Scholes.
2. Joint temporal description of the $(P,V)$ pair.
3. Marked point-process formulation.
4. Explicit buy/sell intensity representation of $dP$ and $dV$.
5. Substitution into the volume-driven SV-Black–Scholes equations.
6. Algebraic inversion for $\lambda^+$ and $\lambda^-$.
7. Extension of the inversion using the full CIR parameters $\kappa,\theta$.
8. Identification formulae for all SV parameters.
9. Extended Kalman Filter implementation for the latent volume factor.

---

*Document generated from the model-development conversation.  
Repository: `nicholashui/price-volume-model`*
