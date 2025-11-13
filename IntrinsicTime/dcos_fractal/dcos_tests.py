# ================================================================
# Purpose: statistical tests for p ≈ 1 - e^{-1} in DcOS results
# ================================================================
import numpy as np
import pandas as pd
from scipy.stats import chisquare, kstest, expon, geom

# ------------------------------------------------
# 1. Estimate p per threshold and compare to 1 - 1/e
# ------------------------------------------------
def estimate_p(results: pd.DataFrame) -> pd.DataFrame:
    """Compute p1 = DC / (DC+OS), p2 = 1/(1+OS/DC) and SEs."""
    df = results.copy()
    df["p1"] = df["nDCtot"] / (df["nDCtot"] + df["nOStot"])
    df["p2"] = 1 / (1 + (df["nOStot"] / df["nDCtot"]))
    df["p_mean"] = 0.5 * (df["p1"] + df["p2"])
    df["p_se"] = np.sqrt(df["p_mean"] * (1 - df["p_mean"]) /
                         np.maximum(df["nDCtot"] + df["nOStot"], 1))
    df["diff_from_const"] = df["p_mean"] - (1 - np.exp(-1))
    return df


# ------------------------------------------------
# 2. Extract OS per run (K) from DcOS event logs
# ------------------------------------------------
def extract_os_counts(event_sequence):
    """
    event_sequence: iterable of emitted codes from DcOS.run()
       convention: DC -> -side (+/-1), OS -> 2*side (+/-2)
    returns list of K (#OS between two DCs)
    """
    K, counter = [], 0
    for ev in event_sequence:
        if ev == 0:
            continue
        if abs(ev) == 2:          # overshoot
            counter += 1
        elif abs(ev) == 1:        # directional change
            K.append(counter)
            counter = 0
    return np.array(K, dtype=int)


# ------------------------------------------------
# 3. Geometric fit test
# ------------------------------------------------
def test_geometric(K):
    """Fit geometric parameter and run χ² + KS tests."""
    if len(K) == 0:
        return np.nan, np.nan, np.nan
    p_hat = 1.0 / (1.0 + np.mean(K))
    # theoretical pmf up to max observed
    vals, counts = np.unique(K, return_counts=True)
    probs = geom.pmf(vals, p_hat)
    probs /= probs.sum()  # normalize finite support
    chi2, p_chi2 = chisquare(counts, f_exp=probs * len(K))
    ks, p_ks = kstest(K, "geom", args=(p_hat,))
    return p_hat, p_chi2, p_ks


# ------------------------------------------------
# 4. Exponential hazard test using osSegment data
# ------------------------------------------------
def test_exponential(os_segments):
    """Fit exponential(λ) to overshoot lengths and test fit."""
    x = np.array(os_segments, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) < 5:
        return np.nan, np.nan, (np.nan, np.nan)
    λ_hat = 1.0 / np.mean(x)
    ks, p_ks = kstest(x, "expon", args=(0, 1 / λ_hat))
    ci95 = (λ_hat * (1 - 1.96 / np.sqrt(len(x))),
            λ_hat * (1 + 1.96 / np.sqrt(len(x))))
    return λ_hat, p_ks, ci95


# ------------------------------------------------
# 5. Combined diagnostic routine
# ------------------------------------------------
def analyze_dcos_results(results, event_sequences=None, os_segments=None):
    """
    results: DataFrame from DcOS_fractal.run_count()
    event_sequences: optional dict {threshold: [event codes]}
    os_segments: optional dict {threshold: [os lengths]}
    Returns enriched summary DataFrame.
    """
    summary = estimate_p(results)
    out = []
    for _, row in summary.iterrows():
        δ = row["threshold"]
        record = {"threshold": δ,
                  "p_mean": row["p_mean"],
                  "diff": row["diff_from_const"]}
        # per-threshold optional deeper tests
        if event_sequences and δ in event_sequences:
            K = extract_os_counts(event_sequences[δ])
            p_hat, p_chi2, p_ks = test_geometric(K)
            record.update({"p_geom": p_hat,
                           "geo_chi2_p": p_chi2,
                           "geo_ks_p": p_ks})
        if os_segments and δ in os_segments:
            λ_hat, p_ks, ci95 = test_exponential(os_segments[δ])
            if isinstance(ci95, (tuple, list)) and len(ci95) == 2:
                ci_low, ci_high = ci95
            else:
                ci_low, ci_high = (np.nan, np.nan)
            record.update({
                "λ_hat": λ_hat,
                "exp_ks_p": p_ks,
                "λ_ci_low": ci_low,
                "λ_ci_high": ci_high,
                "p_pred": 1 - np.exp(-λ_hat if np.isfinite(λ_hat) else np.nan)
            })
        out.append(record)
    return pd.DataFrame(out)
