"""
PIPELINE.md open item #3: Zeliade quasi-explicit SVI calibration.

Raw SVI, per expiry slice: total variance
    w(k) = a + b*(rho*(k-m) + sqrt((k-m)**2 + sigma**2))
where k = log-moneyness = ln(K/F), w = implied_vol**2 * T.

Zeliade trick: fix (m, sigma), substitute y = (k-m)/sigma, and the model
becomes linear in (c1, c2, c3) = (a, b*rho*sigma, b*sigma):
    w = c1 + c2*y + c3*sqrt(y**2 + 1)
That's a 3-parameter linear least-squares (closed form, no local minima) for
any fixed (m, sigma). So the *only* nonlinear search is a stable 2D search
over (m, sigma); (a, b, rho) drop out of the inner solve algebraically. This
avoids the raw 5-parameter fit's local-minima trapping.

No-arbitrage constraints enforced on the fitted slice (Gatheral-Jacquier):
- b >= 0, |rho| <= 1, sigma > 0                      (well-formed SVI)
- butterfly: g(k) >= 0 everywhere                     (non-negative density)
- wings: w(k) <= 2|k| as |k| -> infinity               (Roger Lee moment bound)
Calendar (w non-decreasing in T at fixed k) is checked *across* slices, not
within one, since it needs two expiries.
"""
import numpy as np
from scipy.optimize import minimize


def svi_total_variance(k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray:
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def _inner_linear_fit(y: np.ndarray, w: np.ndarray, weights: np.ndarray, sigma: float):
    """
    Solve w = c1 + c2*y + c3*sqrt(y**2+1) by weighted linear least squares,
    then clip into the no-arbitrage cone (c3 >= 0, |c2| <= c3) and re-solve
    the intercept c1 alone if clipping was needed. Returns (a, b, rho, sse).

    (c1, c2, c3) = (a, b*rho*sigma, b*sigma), so recovering b needs c3/sigma,
    not c3 alone -- sigma is a required argument specifically so this can't
    be gotten wrong by silently dropping the division.
    """
    X = np.column_stack([np.ones_like(y), y, np.sqrt(y ** 2 + 1)])
    sw = np.sqrt(weights)
    Xw, ww = X * sw[:, None], w * sw

    c, *_ = np.linalg.lstsq(Xw, ww, rcond=None)
    c1, c2, c3 = c

    if c3 < 0 or abs(c2) > c3:
        c3 = max(c3, 1e-8)
        c2 = np.clip(c2, -c3, c3)
        # re-fit c1 alone (1D LS) holding c2, c3 fixed, so the intercept
        # still matches the data as closely as possible after clipping
        resid = w - (c2 * y + c3 * np.sqrt(y ** 2 + 1))
        c1 = np.sum(weights * resid) / np.sum(weights)

    a = c1
    b = c3 / sigma
    rho = c2 / c3 if c3 > 1e-12 else 0.0
    rho = np.clip(rho, -1.0, 1.0)

    w_fit = c1 + c2 * y + c3 * np.sqrt(y ** 2 + 1)
    sse = float(np.sum(weights * (w - w_fit) ** 2))
    return a, b, rho, sse


def fit_svi_slice(k: np.ndarray, w: np.ndarray, weights: np.ndarray = None):
    """
    Outer search over (m, sigma); inner (a, b, rho) solved exactly per
    Zeliade above. Returns dict of fitted params + fit diagnostics.
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    if weights is None:
        weights = np.ones_like(k)
    weights = np.asarray(weights, dtype=float)

    def outer_objective(params):
        m, log_sigma = params
        sigma = np.exp(log_sigma)  # optimize in log-space, sigma must stay > 0
        y = (k - m) / sigma
        _, _, _, sse = _inner_linear_fit(y, w, weights, sigma)
        return sse

    m0 = float(np.average(k, weights=weights))
    k_span = float(k.max() - k.min())
    sigma0 = max(float(np.std(k)), 0.05)

    # Bound (m, sigma) to the physically sane region implied by the data's
    # own log-moneyness range. Unconstrained, the outer search can wander to
    # a degenerate (m, sigma) far outside the quoted strikes when there are
    # few points -- it finds a curve that threads the sparse data exactly
    # but is nonsense as a smile (seen in practice: rho pinned to +/-1, m
    # miles outside the strike range). Bounds keep the search in the region
    # the data can actually inform.
    m_bounds = (k.min() - k_span, k.max() + k_span)
    log_sigma_bounds = (np.log(max(k_span * 0.02, 1e-3)), np.log(max(k_span * 3, 0.5)))

    result = minimize(
        outer_objective, x0=[m0, np.log(sigma0)],
        method="L-BFGS-B",
        bounds=[m_bounds, log_sigma_bounds],
    )
    m, sigma = result.x[0], np.exp(result.x[1])
    y = (k - m) / sigma
    a, b, rho, sse = _inner_linear_fit(y, w, weights, sigma)

    return {
        "a": a, "b": b, "rho": rho, "m": m, "sigma": sigma,
        "sse": sse, "n_points": len(k), "converged": result.success,
    }


def check_butterfly_arbitrage(params: dict, k_grid: np.ndarray) -> np.ndarray:
    """
    Gatheral-Jacquier g(k) >= 0 condition. g < 0 anywhere means the SVI
    slice implies a negative density there -- must not happen.
    """
    a, b, rho, m, sigma = params["a"], params["b"], params["rho"], params["m"], params["sigma"]
    w = svi_total_variance(k_grid, a, b, rho, m, sigma)
    dk = k_grid[1] - k_grid[0]
    wp = np.gradient(w, dk)
    wpp = np.gradient(wp, dk)

    g = (1 - k_grid * wp / (2 * w)) ** 2 - (wp ** 2 / 4) * (1 / w + 0.25) + wpp / 2
    return g


def check_wing_bound(params: dict, k_grid: np.ndarray) -> np.ndarray:
    """Roger Lee moment bound: w(k) <= 2|k| as |k| -> large. Returns slack
    (positive = bound respected, negative = violated)."""
    a, b, rho, m, sigma = params["a"], params["b"], params["rho"], params["m"], params["sigma"]
    w = svi_total_variance(k_grid, a, b, rho, m, sigma)
    return 2 * np.abs(k_grid) - w


def check_calendar_arbitrage(params_near: dict, params_far: dict, k_grid: np.ndarray) -> np.ndarray:
    """Total variance must be non-decreasing in T at fixed k. Returns slack
    (w_far - w_near; must be >= 0 everywhere)."""
    w_near = svi_total_variance(k_grid, **{kk: params_near[kk] for kk in ("a", "b", "rho", "m", "sigma")})
    w_far = svi_total_variance(k_grid, **{kk: params_far[kk] for kk in ("a", "b", "rho", "m", "sigma")})
    return w_far - w_near

def svi_w_func(params: dict):
    """Wraps a single fitted SVI slice as w(k) -> total variance, callable at
    any k (not just a precomputed grid) -- just the closed-form formula with
    the 5 fitted params bound in."""
    keys = ("a", "b", "rho", "m", "sigma")
    values = [params[kk] for kk in keys]
    return lambda k: svi_total_variance(k, *values)


def make_blended_w_func(params_near: dict, params_far: dict, alpha: float):
    """
    Returns w(k) -> total variance, blending two fitted SVI slices via
    linear interpolation in total variance at fixed log-moneyness k. Each
    call does two exact closed-form SVI evaluations and combines them
    algebraically -- no grid, no spline, callable at any k including the
    tiny K +/- dK steps a finite-difference density calc needs.

    alpha is the fixed calendar-time weight toward the target date:
        alpha = (T_target - T_near) / (T_far - T_near)
    Compute it once from the three dates before calling this -- it's a
    single scalar for the whole curve, not something that varies with k.
    """
    w_near = svi_w_func(params_near)
    w_far = svi_w_func(params_far)
    return lambda k: (1 - alpha) * w_near(k) + alpha * w_far(k)



