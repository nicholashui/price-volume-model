# price-volume-model

Complete documentation of a price–volume stochastic model that starts from classical Black–Scholes and progressively incorporates:

- Joint \((P, V)\) dynamics
- Marked point processes with buy / sell intensities
- Volume-driven stochastic volatility (CIR factor)
- Algebraic recovery of intensities from price & volume
- Parameter identification
- Extended Kalman Filter for the latent volume factor

See **[model.md](model.md)** for the full derivation, all equations, and the reference Python EKF implementation.

Repository created and documented from an interactive modelling session.
