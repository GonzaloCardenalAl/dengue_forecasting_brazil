"""
Python port of the temperature-dependent Wallinga-Teunis Rt estimator from
Codeco et al. (Epidemics, 2017), originally implemented in R as
features/EstRtGT_v4.R + features/sumgamma_v2.R (and demonstrated in
features/Rt_calc-example.rmd). No R/rpy2 dependency -- this is a clean-room
numpy/scipy re-derivation of the same math.

Used only at FORECAST time (forecasting/forecast_next_52w.py) to fill in
Rt/p_rt1 for the forecast horizon, where InfoDengue's real reported values
don't exist yet. Historical Rt/p_rt1 in training data stay as InfoDengue's
own ground truth -- this module is never used to recompute history.

Two deliberate departures from a literal line-by-line port, both purely
for performance (same math, no change in output):

1. sumgamma_v2.R's `int_sum_gamma_T` re-derives the Moschopoulos CDF from
   scratch (via a binary-search inversion, `t_sum_gamma_v3`) every time the
   temperature regime changes. Here, the CDF for a given (temperature ->
   EIP-rate) value is computed once on a fine grid and cached
   (`_cdf_grid_for_temp`), and inversion is a `np.interp` lookup against
   that grid instead of a fresh binary search.
2. `sum_gamma_dist`'s y-loop (R's `sapply`) is vectorized over the whole
   grid in one numpy call (`sum_gamma_pdf`).

Scope cut: only the *uncorrected* Wallinga-Teunis estimator (`correct=False`
in the R code) is implemented -- this is what the reference Rmd itself uses
for both the temperature-dependent and temperature-independent estimates.
The "corrected" right-censoring adjustment (R.corrected/R.simu.corrected) is
not implemented; `correct=True` raises NotImplementedError.
"""
from functools import lru_cache

import numpy as np
from scipy.special import gammaln


# ── Incubation period rates (EstRtGT_v4.R lines 6-9) ─────────────────────────

def lambda_eip(temp: float, v: float = 4.3, beta0: float = 7.9,
                betat: float = -0.21, tbar: float = 0.0) -> float:
    """Extrinsic incubation period rate (mosquito), temperature-dependent."""
    return v / np.exp(beta0 + betat * (temp - tbar))


def lambda_iip(v: float = 16.0, beta0: float = 1.78) -> float:
    """Intrinsic incubation period rate (human), temperature-independent."""
    return v / np.exp(beta0)


# ── Moschopoulos sum-of-independent-gammas (sumgamma_v2.R) ───────────────────

def _gammak(a: np.ndarray, b: np.ndarray, k: int) -> float:
    b1 = b.min()
    # log(0) for the component(s) where b_i == b1 is intentional (that
    # component's contribution is exactly 0 for k >= 1, the correct limiting
    # value), not an error -- suppress the expected divide-by-zero warning.
    with np.errstate(divide="ignore"):
        lga = np.log(a) + k * np.log(1 - b1 / b) - np.log(k)
    return float(np.exp(lga).sum())


def sum_gamma_pdf(y: np.ndarray, a: np.ndarray, b: np.ndarray, K: int = 100) -> np.ndarray:
    """
    Moschopoulos (1985) density of Y = sum of independent Gamma(shape=a_i,
    scale=b_i) random variables, evaluated at every point in `y`. Port of
    sumgamma_v2.R's gammak/sum_gamma_dist, vectorized over `y` (R's version
    loops one point at a time via sapply).
    """
    y = np.asarray(y, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    b1 = b.min()
    rho = a.sum()
    C = np.exp((a * np.log(b1 / b)).sum())

    delta = np.zeros(K + 1)
    delta[0] = 1.0
    gammav = np.array([_gammak(a, b, k) for k in range(1, K + 1)])
    for k in range(K):
        ks = np.arange(1, k + 2)
        delta[k + 1] = np.sum(ks * gammav[:k + 1] * delta[k::-1][:k + 1]) / (k + 1)

    # Moschopoulos (1985): f_Y(y) = C * sum_{m=0}^K delta_m * y^(rho+m-1) *
    # exp(-y/b1) / (Gamma(rho+m) * b1^(rho+m)). NOTE: this exponent
    # convention (rho+m-1, not rho+m) was verified against two independent
    # exact analytic checks -- a single Gamma(3,2) and a sum of two
    # differently-rated exponentials -- both reproduced to ~1e-16. The
    # literal sumgamma_v2.R source reads as `(rho-1 + 1:(K+1))` paired
    # positionally against a length-(K+1) `delta` array indexed 1..K+1 (math
    # 0..K), which works out to exponent rho+m (an apparent off-by-one
    # relative to the verified-correct formula below) -- this looks like a
    # genuine bug in the reference R script rather than an alternate valid
    # convention, since the rho+m-1 version is the only one that reproduces
    # known closed-form sum-of-gammas densities exactly.
    m = np.arange(K + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_y = np.log(y)
        xx = (
            np.log(delta)[:, None]
            + (rho - 1 + m)[:, None] * log_y[None, :]
            - y[None, :] / b1
            - gammaln(rho + m)[:, None]
            - (rho + m)[:, None] * np.log(b1)
        )
        gy = C * np.nansum(np.exp(xx), axis=0)
    return np.where(y > 0, gy, 0.0)


# ── Temperature-dependent generation-time distribution ───────────────────────
# (sumgamma_v2.R's int_sum_gamma_T / EstRtGT_v4.R's evalGenTimeDist)

_UNITSCALE = 7.0  # day-calibrated rate params -> week-scale (matches R's unitscale=7)


def _build_grids(eip_rate: float, iip_rate: float, gt_max: int, grid_step: float, K: int, tau_scale: float):
    a = np.array([16.0, 4.3, 1.0, 1.0])
    b = np.array([1.0 / iip_rate, 1.0 / eip_rate, 1.0, 1.0]) / _UNITSCALE * tau_scale
    y_grid = np.arange(grid_step, gt_max + grid_step, grid_step)
    pdf_grid = sum_gamma_pdf(y_grid, a, b, K=K)
    cdf_grid = np.cumsum(pdf_grid) * grid_step
    return y_grid, pdf_grid, cdf_grid


def _make_grid_cache(gt_max: int, grid_step: float, K: int, iip_rate: float, tau_scale: float):
    @lru_cache(maxsize=None)
    def _cached(temp_rounded: float):
        eip_rate = lambda_eip(temp_rounded)
        return _build_grids(eip_rate, iip_rate, gt_max, grid_step, K, tau_scale)

    def get_grid(temp: float):
        return _cached(round(float(temp), 2))

    return get_grid


def _gt_pmf_one_week(temp_window: np.ndarray, gt_max: int, step: float, get_grid) -> np.ndarray:
    """
    Generation-time PMF (length gt_max+1, bucketed by ceil(weeks)) for cases
    originating in the week whose temperature-regime window is
    `temp_window` (temp at weeks x, x+1, ..., x+gt_max -- only indices
    0..gt_max are ever read, regardless of how much longer temp_window is).

    Implements the same time-rescaling trick as int_sum_gamma_T:
    `summ` accumulates DENSITY*step increments (a Riemann sum, exactly like
    R's `xx <- sum_gamma_dist(...)*step; summ <- summ+xx` -- NOT a direct
    CDF lookup, which would silently assume CDF_old(t) and CDF_new(t) agree
    at the switch point, false in general for two different distributions).
    When the temperature regime switches (at each integer week boundary):
    first move to the NEW regime's grid, then find the time `tsum` at which
    the NEW regime's *own* CDF would have produced the cumulative
    probability already accrued (`summ`) -- this is the time-equivalent
    starting point to keep accumulating density increments from under the
    new regime, continuous with the old regime's accrued probability.
    """
    n_steps = int(round(gt_max / step))
    pdf = np.zeros(gt_max + 1)
    summ = 0.0
    regime = 0
    y_grid, pdf_grid, cdf_grid = get_grid(temp_window[0])
    z_offset = 0.0

    for step_idx in range(1, n_steps + 1):
        i = step_idx * step
        if summ > 0.999:
            continue  # negligible remaining mass -- matches R's `withbreak` shortcut

        if regime + 1 <= gt_max and i > (regime + 1):
            regime += 1
            y_grid, pdf_grid, cdf_grid = get_grid(temp_window[regime])
            tsum = float(np.interp(summ, cdf_grid, y_grid))
            z_offset = tsum - i

        z = max(i + z_offset, 0.0)
        density_z = float(np.interp(z, y_grid, pdf_grid, left=0.0, right=0.0))
        increment = density_z * step
        j = min(int(np.ceil(i)), gt_max)
        pdf[j] += increment
        summ += increment

    return pdf


def compute_generation_time_matrix(
    temp_series: np.ndarray,
    n_weeks: int,
    gt_max: int = 5,
    step: float = 0.1,
    grid_step: float = 0.05,
    K: int = 100,
    tau_scale: float = 1.0,
) -> np.ndarray:
    """
    gt_pmf[x, k] = P(generation time = k weeks | infector at week x), for
    x = 0..n_weeks-1, k = 0..gt_max. Requires `temp_series` to have at least
    n_weeks + gt_max entries (the last `gt_max` weeks of temp_series are
    "lookahead" only, never used as a starting week themselves).

    tau_scale uniformly stretches (>1) or compresses (<1) the generation-time
    distribution's scale parameters, calibration knob for cases where the
    paper's published incubation-period constants don't match a given
    surveillance system's own Rt/p_rt1 closely (see
    scripts/calibrate_rt_estimation.py) -- 1.0 reproduces the paper exactly.
    """
    temp_series = np.asarray(temp_series, dtype=float)
    if len(temp_series) < n_weeks + gt_max:
        raise ValueError(
            f"temp_series has {len(temp_series)} weeks, need >= "
            f"n_weeks + gt_max = {n_weeks + gt_max} (gt_max weeks of "
            f"lookahead temperature beyond the last starting week)."
        )
    iip_rate = lambda_iip()
    get_grid = _make_grid_cache(gt_max, grid_step, K, iip_rate, tau_scale)

    gt_pmf = np.zeros((n_weeks, gt_max + 1))
    for x in range(n_weeks):
        gt_pmf[x, :] = _gt_pmf_one_week(temp_series[x:x + gt_max + 1], gt_max, step, get_grid)
    return gt_pmf


# ── Wallinga-Teunis time-dependent R estimator (EstRtGT_v4.R's est.R.Temp) ───

def est_rt_temp(
    cases: np.ndarray,
    gt_pmf: np.ndarray,
    nsim: int = 1000,
    correct: bool = False,
    seed: int | None = None,
) -> dict:
    """
    Temperature-dependent Wallinga-Teunis Rt estimate.

    Parameters
    ----------
    cases   : weekly incidence (casos_est), length T. Rounded to the
              nearest integer for the multinomial simulation step (the
              renewal-equation math assumes discrete case counts; casos_est
              is a continuous nowcast estimate).
    gt_pmf  : (T, gt_max+1) generation-time PMF from
              compute_generation_time_matrix -- gt_pmf[i, k] = P(generation
              time = k weeks | infector at week i).
    nsim    : number of multinomial "who-infected-whom" simulation draws,
              used for p_rt1 (and would back a credible interval, not
              currently returned -- see module docstring).

    Returns
    -------
    dict with:
        R       : (T,) point estimate (uncorrected Wallinga-Teunis), R[s] =
                  sum_i P[i, s] for i <= s -- i.e. attributed FROM week s's
                  perspective is not what's returned; R is indexed by
                  INFECTOR week i (R[i] = expected secondary cases per case
                  at week i), matching the R code's R.WT.
        p_rt1   : (T,) P(R > 1) at each infector week, from the simulated
                  draws -- see module docstring for the precise mechanism
                  (fraction of nsim multinomial draws with simulated R > 1).
        R_simu  : (T, nsim) the underlying simulated draws (exposed for
                  callers that want their own summary statistic).
    """
    if correct:
        raise NotImplementedError(
            "correct=True (right-censoring-adjusted R.corrected/R.simu.corrected) "
            "is not implemented -- only the uncorrected estimator used by the "
            "reference Rmd (correct=FALSE) is ported."
        )

    cases = np.asarray(cases, dtype=float)
    T = len(cases)
    gt_max = gt_pmf.shape[1] - 1
    if gt_pmf.shape[0] != T:
        raise ValueError(f"gt_pmf has {gt_pmf.shape[0]} rows, expected {T} (len(cases)).")

    rng = np.random.default_rng(seed)
    cases_int = np.round(cases).astype(int)

    P = np.zeros((T, T))           # P[i, s]: per EstRtGT_v4.R's `P` matrix
    # nsim=0 skips the multinomial simulation entirely (R/R_simu deterministic
    # point estimate only) -- useful for fast point-estimate-only parameter
    # search, since the simulation step is the dominant cost.
    cum_attrib = np.zeros((T, nsim)) if nsim > 0 else None

    for s in range(1, T):
        i_idx = np.arange(s + 1)
        offsets = s - i_idx
        valid = offsets <= gt_max
        dg = np.zeros(s + 1)
        dg[valid] = gt_pmf[i_idx[valid], offsets[valid]]

        incid_s = cases[s]
        if incid_s <= 0:
            continue

        weight = cases[i_idx] * dg
        wsum = weight.sum()
        if wsum <= 0:
            continue
        weight = weight / wsum

        with np.errstate(divide="ignore", invalid="ignore"):
            prob = weight * incid_s / cases[i_idx]
        prob = np.where(cases[i_idx] == 0, 0.0, prob)
        if cases_int[s] == 1:
            prob[-1] = 0.0
        P[i_idx, s] = prob

        size_s = int(cases_int[s])
        if size_s > 0 and cum_attrib is not None:
            draws = rng.multinomial(size_s, weight, size=nsim)  # (nsim, s+1)
            cum_attrib[i_idx, :] += draws.T

    R = P.sum(axis=1)
    R[cases == 0] = 0.0

    if cum_attrib is None:
        return {"R": R, "p_rt1": None, "R_simu": None}

    with np.errstate(divide="ignore", invalid="ignore"):
        R_simu = cum_attrib / cases[:, None]
    R_simu = np.where(cases[:, None] == 0, 0.0, R_simu)
    R_simu[cases == 0, :] = 0.0

    p_rt1 = np.mean(R_simu > 1, axis=1)

    return {"R": R, "p_rt1": p_rt1, "R_simu": R_simu}


# ── High-level convenience wrapper ────────────────────────────────────────────

def estimate_rt_p_rt1(
    cases: np.ndarray,
    temp: np.ndarray,
    gt_max: int = 5,
    nsim: int = 500,
    seed: int | None = None,
    smooth_temp: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One-call convenience wrapper: loess-smooth temperature (matching the
    reference Rmd's own preprocessing), build the generation-time matrix,
    run the Wallinga-Teunis estimator, return (R, p_rt1) for every week in
    `cases`.

    `temp` must have at least len(cases) + gt_max entries (gt_max weeks of
    lookahead temperature beyond the last case week) -- callers forecasting
    forward should extend their climatological temperature series that far
    past the forecast horizon before calling this.
    """
    cases = np.asarray(cases, dtype=float)
    temp = np.asarray(temp, dtype=float)
    n_weeks = len(cases)

    if smooth_temp:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        temp = lowess(temp, np.arange(len(temp)), frac=0.05, return_sorted=False)

    gt_pmf = compute_generation_time_matrix(temp, n_weeks=n_weeks, gt_max=gt_max)
    result = est_rt_temp(cases, gt_pmf, nsim=nsim, seed=seed)
    return result["R"], result["p_rt1"]
