import pandas as pd
import numpy as np
from scipy.stats import linregress
import os
from pathlib import Path
import pickle
from concurrent.futures import ProcessPoolExecutor

from dcos_core.dcos_core import DcOS, Sample


class DcOS_fractal:
    """
    Intrinsic Time fractal scaling analysis.
    Final regression region:
      - lower bound: index of first δ where %DC ≥ low_pt%
      - upper bound: index of δ, greater than lower bound just before first δ where low_pt-high_pt_change < %DC < low_pt+high_pt_change
    Adds predicted y values (y_pred_*) for all fitted frequencies.
    """

    def __init__(self, thresholds=None, initialMode=0, debugMode=False): # threshWinLen=7, r2min=0.98,
        if thresholds is None:
            thresholds = np.logspace(-5, 0, 100)
        self.thresholds = thresholds
        self.initialMode = initialMode
        self.debugMode = debugMode
        # self.threshWinLen = threshWinLen
        # self.r2min = r2min

        # self.df = None
        # self.dfPath = None


    # ------------------------------ Input Validation ------------------------------
    @staticmethod
    def _validate_input(df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        if not {"Timestamp", "Price"}.issubset(df.columns):
            raise ValueError("Input DataFrame must contain columns ['Timestamp', 'Price'].")
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        if not np.issubdtype(df["Price"].dtype, np.number):
            raise TypeError("Column 'Price' must be numeric.")
        return True


    # ------------------------------ Single threshold run ------------------------------
    @staticmethod
    def _run_single_threshold(args):
        δ, df, initialMode = args
        dcos = DcOS(threshold=δ, initialMode=initialMode, midpriceMode=False)
        for _, row in df.iterrows():
            dcos.run(Sample(row["Price"], row["Timestamp"]))
        return δ, dcos.nDCtot, dcos.nOStot, dcos.nDCtot + dcos.nOStot


    def run_dcos_counts_parallel(self, df, thresholds=None, initialMode=0, max_workers=None):
        self._validate_input(df)
        if thresholds is None:
            thresholds = self.thresholds
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(self._run_single_threshold,
                                  [(δ, df, initialMode) for δ in thresholds]))
        return pd.DataFrame(results, columns=["threshold", "nDCtot", "nOStot", "nEVtot"])


    def run_dcos_counts(self, df, thresholds=None, initialMode=0):
        self._validate_input(df)
        if thresholds is None:
            thresholds = self.thresholds
        data = []
        for δ in thresholds:
            dcos = DcOS(threshold=δ, initialMode=initialMode, midpriceMode=False)
            for _, row in df.iterrows():
                sample = Sample(row["Price"], row["Timestamp"])
                dcos.run(sample)
            data.append((δ, dcos.nDCtot, dcos.nOStot, dcos.nDCtot + dcos.nOStot))
        return pd.DataFrame(data, columns=["threshold", "nDCtot", "nOStot", "nEVtot"])


    # ------------------------------ Frequency Computation ------------------------------
    def compute_freqs(self, results, n_ticks):
        for key in ["nDCtot", "nOStot", "nEVtot"]:
            results[f"{key}_freq"] = results[key] / n_ticks
            p = results[f"{key}_freq"]
            results[f"{key}_stderr"] = np.sqrt(p * (1 - np.minimum(p, 1)) / n_ticks)
        results["dc_ratio"] = results["nDCtot_freq"] / results["nEVtot_freq"]
        results["dc_pct"] = 100 * results["dc_ratio"]

        p = results["dc_ratio"]
        n = results["nEVtot"]
        results["dc_pct_stderr"] = 100 * np.sqrt(p * (1 - p) / np.maximum(n, 1))
        return results


    # ------------------------------ Fit Region: centered on low_pt ------------------------------
    def determine_fit_region(self, results, low_pt=61.5, high_pt_change=4):
        dc = results["dc_pct"].values
        δ  = results["threshold"].values
        n  = len(dc)
        if n == 0:
            results.attrs["δ_min_fit"] = np.nan
            results.attrs["δ_max_fit"] = np.nan
            return results

        # lower bound: first index where %DC crosses low_pt
        if dc[0] < low_pt:
            idx_low = np.argmax(dc > low_pt) if np.any(dc > low_pt) else None
        else:
            idx_low = np.argmax(dc <  low_pt + 1.5) if np.any(dc <  low_pt + 1.5) else None
        if idx_low is None or idx_low == 0:
            results.attrs["δ_min_fit"] = np.nan
            results.attrs["δ_max_fit"] = np.nan
            return results

        δ_min_fit = δ[idx_low]

        # upper bound: first δ > idx_low where %DC leaves the band
        lo = low_pt - high_pt_change
        hi = low_pt + high_pt_change
        δ_max_fit = np.nan
        idx_high  = idx_low
        for j in range(idx_low + 1, n):
            if not (lo < dc[j] < hi):        # ← leave band
                idx_high  = j - 1            # last in-band point
                δ_max_fit = δ[idx_high]
                break
        if not np.isfinite(δ_max_fit):
            idx_high  = n - 1
            δ_max_fit = δ[idx_high]

        results.attrs["idx_low"]   = idx_low
        results.attrs["idx_high"]  = idx_high
        results.attrs["δ_min_fit"] = δ_min_fit
        results.attrs["δ_max_fit"] = δ_max_fit
        return results


    # ------------------------------ Tail Fitting ------------------------------
    def analyze_tail_scaling(self, results, δ_min_fit=None, δ_max_fit=None):
        """Fit regression within the δ_min_fit – δ_min_fit% region+-δ_max_fit."""
        if δ_min_fit is None:
            δ_min_fit = results.attrs.get("δ_min_fit", np.nan)
        if δ_max_fit is None:
            δ_max_fit = results.attrs.get("δ_max_fit", np.nan)

        if not np.isfinite(δ_min_fit) or not np.isfinite(δ_max_fit):
            if self.debugMode:
                print("Invalid fit region → skipping regression.")
            return results

        mask = (results["threshold"] >= δ_min_fit) & (results["threshold"] <= δ_max_fit)
        trimmed = results.loc[mask].copy()

        if len(trimmed) < 2:
            if self.debugMode:
                print("Not enough points for regression fit.")
            return results

        fits = {}
        for key in ["nEVtot_freq", "nDCtot_freq", "nOStot_freq"]:
            mask_valid = trimmed[key] > 0
            x = np.log10(trimmed.loc[mask_valid, "threshold"].values)
            y = np.log10(trimmed.loc[mask_valid, key].values)
            if len(np.unique(x)) < 2:
                fits[key] = {"slope": np.nan, "intercept": np.nan, "r2": np.nan}
                results[f"y_pred_{key}"] = np.nan
                continue
            slope, intercept, r, _, _ = linregress(x, y)
            if np.isnan(slope) or np.isnan(intercept):
                if self.debugMode:
                    print(f"Skipping {key} fit due to NaNs.")
                results[f"y_pred_{key}"] = np.nan
                continue
            fits[key] = {"slope": slope, "intercept": intercept, "r2": r**2}
            results[f"y_pred_{key}"] = 10 ** (intercept + slope * np.log10(results["threshold"]))

        results.attrs["tail_fit"] = fits
        results.attrs["fit_region"] = {"δ_min_fit": δ_min_fit, "δ_max_fit": δ_max_fit}

        if self.debugMode:
            for k, v in fits.items():
                print(f"{k}: β={v['slope']:.3f}, R²={v['r2']:.3f}")
        return results


    # ------------------------------ Main Pipeline ------------------------------
    def run_count(self, df=None, dfPath=None, dfName=None, parallel=True): # , low_pt=61.5, high_pt_change=4,
        if df is None:
            if not dfName:
                raise ValueError("Provide either a DataFrame or dfName.")
            ext = Path(dfName).suffix.lower()
            full_path = Path(dfPath or ".") / dfName
            df = pd.read_csv(full_path) if ext == ".csv" else pd.read_parquet(full_path)

        # self.df, self.dfPath = df, dfPath or os.getcwd()
        results = self.run_dcos_counts_parallel(df) if parallel else self.run_dcos_counts(df)
        results = self.compute_freqs(results, len(df))
        results.attrs["thresholds"] = self.thresholds
        # results = self.run_analysis(dfPath, dfName, low_pt, high_pt_change)
        return results


    def run_analysis(self, results, low_pt=61.5, high_pt_change=4):
        # results = self.load_results(dfPath, dfName)
        results = self.determine_fit_region(results, low_pt, high_pt_change)
        results = self.analyze_tail_scaling(results)
        # results.attrs.update({
        #     "params": {
        #         "thresholds": self.thresholds.tolist(),
        #         "threshWinLen": self.threshWinLen,
        #         "r2min": self.r2min,
        #     },
        # })
        return results


    def run_count_and_analysis(self, df=None, dfPath=None, dfName=None, low_pt=61.5, high_pt_change=4, parallel=True):
        if df is None:
            if not dfName:
                raise ValueError("Provide either a DataFrame or dfName.")
            ext = Path(dfName).suffix.lower()
            full_path = Path(dfPath or ".") / dfName
            df = pd.read_csv(full_path) if ext == ".csv" else pd.read_parquet(full_path)

        results = self.run_count(df, dfPath, dfName, parallel)
        results = self.run_analysis(results, low_pt, high_pt_change)
        return results


    # ------------------------------ Save / Load ------------------------------
    def save_results(self, results, dfPath=None, dfName="dcos_results.pkl"):
        self.dfPath = dfPath
        path = Path(self.dfPath or ".") / dfName
        with open(path, "wb") as f:
            pickle.dump(results, f)
        if self.debugMode:
            print(f"Saved results to {path}")
        return self

    def load_results(self, dfPath=None, dfName="dcos_results.pkl"):
        self.dfPath = dfPath
        path = Path(self.dfPath or ".") / dfName
        if not path.exists():
            raise FileNotFoundError(f"No pickle file found at {path}")
        with open(path, "rb") as f:
            results = pickle.load(f)
        if self.debugMode:
            print(f"Loaded results from {path}")
        return results
