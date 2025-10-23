import pandas as pd
import numpy as np
from scipy.stats import linregress
import os
from pathlib import Path
from dcos_core.dcos_core import DcOS, Sample

class DcOS_fractal:
    def __init__(self, thresholds=None, threshWinLen=7, r2min=0.98, initialMode=-1, debugMode=False):
        if thresholds is None:
            thresholds = np.logspace(-5, -1, 30)
        self.thresholds = thresholds
        self.threshWinLen = threshWinLen
        self.r2min = r2min
        self.initialMode = initialMode
        self.debugMode = debugMode
        self.df = None
        self.dfPath = None

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

    def run_dcos_counts(self, df, thresholds=None, initialMode=None):
        self._validate_input(df)
        if thresholds is None:
            thresholds = self.thresholds
        if initialMode is None:
            initialMode = self.initialMode

        data = []
        for δ in thresholds:
            dcos = DcOS(threshold=δ, initialMode=initialMode, midpriceMode=False)
            for _, row in df.iterrows():
                sample = Sample(row["Price"], row["Timestamp"])
                dcos.run(sample)
            data.append((δ, dcos.nDCtot, dcos.nOStot, dcos.nDCtot + dcos.nOStot))
        return pd.DataFrame(data, columns=["threshold", "nDCtot", "nOStot", "nEVtot"])

    def compute_freqs(self, results, n_ticks):
        for key in ["nDCtot", "nOStot", "nEVtot"]:
            results[f"{key}_freq"] = results[key] / n_ticks
            p = results[f"{key}_freq"]
            results[f"{key}_stderr"] = np.sqrt(p * (1 - np.minimum(p, 1)) / n_ticks)
        return results

    def fractal_ranges(self, thresholds, freqs):
        δ, f = np.array(thresholds), np.array(freqs)
        mask = f > 0
        δ, f = δ[mask], f[mask]
        x, y = np.log10(δ), np.log10(f)
        ranges = []
        for i in range(len(x) - self.threshWinLen + 1):
            xi, yi = x[i:i+self.threshWinLen], y[i:i+self.threshWinLen]
            if not np.all(np.diff(yi) <= 0):
                continue
            slope, intercept, r, _, _ = linregress(xi, yi)
            if r**2 >= self.r2min:
                ranges.append((10**x[i], 10**x[i+self.threshWinLen-1], slope, r**2))
        return pd.DataFrame(ranges, columns=["δ_L", "δ_U", "slope", "R2"])

    def estimate_breakpoint(self, results, w=None, r2min=None, z=2.0):
        if w is None: w = self.threshWinLen
        if r2min is None: r2min = self.r2min

        th, f = results["threshold"].values, results["nEVtot_freq"].values
        mask = f > 0
        x, y = np.log10(th[mask]), np.log10(f[mask])

        slopes, stderr, centers = [], [], []
        for i in range(len(x) - w + 1):
            xi, yi = x[i:i+w], y[i:i+w]
            slope, _, r, _, s = linregress(xi, yi)
            slopes.append(slope)
            stderr.append(s if np.isfinite(s) else np.nan)
            centers.append(10**x[i + w // 2])

        slopes, stderr, centers = np.array(slopes), np.array(stderr), np.array(centers)
        q75 = np.quantile(centers, 0.75)
        ref = slopes[centers >= q75]
        s_ref = np.median(ref)
        tol = z * np.nanmedian(stderr)
        good = np.abs(slopes - s_ref) <= tol

        first_bad = np.argmax(~good[::-1])
        δ_break = centers[-(first_bad + 1)] if first_bad > 0 else np.nan
        f_break = float(results.loc[(np.abs(results["threshold"] - δ_break)).argmin(), "nEVtot_freq"]) if np.isfinite(δ_break) else np.nan
        return δ_break, f_break

    def run_analysis(self, df=None, dfPath=None, dfName=None):
        if df is None:
            if not dfName:
                raise ValueError("Provide either a DataFrame or dfName.")
            ext = Path(dfName).suffix.lower()
            full_path = Path(dfPath or ".") / dfName
            df = pd.read_csv(full_path) if ext == ".csv" else pd.read_parquet(full_path)

        self.df, self.dfPath = df, dfPath or os.getcwd()

        results = self.run_dcos_counts(df)
        results = self.compute_freqs(results, len(df))
        ranges = self.fractal_ranges(results["threshold"], results["nEVtot_freq"])
        δ_break, f_break = self.estimate_breakpoint(results)
        return results, ranges, δ_break, f_break
