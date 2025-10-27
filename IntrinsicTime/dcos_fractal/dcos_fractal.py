import pandas as pd
import numpy as np
from scipy.stats import linregress
import os
from pathlib import Path
import pickle

from dcos_core.dcos_core import DcOS, Sample


class DcOS_fractal:
    def __init__(self, thresholds=None, threshWinLen=7, r2min=0.98, initialMode=0, debugMode=False):
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


    def analyze_tail_scaling(self, results):
        """Find last valid threshold and fit tail region in log-log space."""
        # valid where 0 < %DC < 100
        dc_pct = 100 * results["nDCtot_freq"] / results["nEVtot_freq"]
        valid_mask = (dc_pct > 0) & (dc_pct < 100)
        last_valid = valid_mask[::-1].idxmax()

        # tail region = last threshWinLen points
        fit_slice = results.iloc[max(0, last_valid - self.threshWinLen + 1): last_valid + 1]
        fits = {}
        for key in ["nEVtot_freq", "nDCtot_freq", "nOStot_freq"]:
            x = np.log10(fit_slice["threshold"].values)
            y = np.log10(fit_slice[key].values)
            slope, intercept, r, _, _ = linregress(x, y)
            fits[key] = {"slope": slope, "intercept": intercept, "r2": r**2}

        results.attrs["tail_fit"] = fits
        results.attrs["last_valid_idx"] = last_valid
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


    def find_upper_cutoff(self, results, threshWinLen=None, weightFactor=4):
        """Find δ_upper dynamically using frequency-weighted R² sensitivity."""
        if threshWinLen is None:
            threshWinLen = self.threshWinLen
        δ = results["threshold"].values
        f = results["nEVtot_freq"].values
        n_ev = results["nEVtot"].values
        x, y = np.log10(δ), np.log10(f)

        # Sliding R² computation
        R2_series = []
        for i in range(len(x) - threshWinLen + 1):
            xi, yi = x[i:i + threshWinLen], y[i:i + threshWinLen]
            _, _, r, _, _ = linregress(xi, yi)
            R2_series.append(r**2)
        R2_series = np.array(R2_series)

        # Normalize weights: higher nEVtot → tighter tolerance
        weights = n_ev[threshWinLen - 1:] / np.max(n_ev)
        weights = np.clip(weights, 1e-3, 1.0)  # avoid zeros

        # Adaptive tolerance: base ± weighted term
        base_tol = 0.01 * np.mean(R2_series)         # base sensitivity
        weighted_tol = base_tol * np.exp(weightFactor * (1 - weights))
        # weighted_tol = base_tol * (1 / weights) ** weightFactor # looser for fewer events

        # diff = np.maximum(0, self.r2min - R2_series)
        diff = np.maximum(0, 1 - R2_series)

        # Identify degradation where diff exceeds adaptive tolerance
        bad_mask = diff > weighted_tol
        bad_idx = np.argmax(bad_mask)

        print(f"weighted_tol = {weighted_tol}")
        print(f"diff = {diff}")

        # if bad_idx == 0 or not np.any(bad_mask):
        #     results.attrs["delta_upper_cutoff"] = np.nan
        #     print("No clear upper cutoff detected.")
        #     return np.nan

        δ_upper = δ[bad_idx + threshWinLen - 1]
        results.attrs["delta_upper_cutoff"] = δ_upper
        print(f"Weighted upper cutoff detected at δ = {δ_upper:.3e}")
        return δ_upper


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
        results = self.analyze_tail_scaling(results)
        ranges = self.fractal_ranges(results["threshold"], results["nEVtot_freq"])
        δ_break, f_break = self.estimate_breakpoint(results)
        δ_upper = self.find_upper_cutoff(results)
        return results, ranges, δ_break, f_break, δ_upper


    def save_results(self, results, ranges, delta_break, f_break, delta_upper, filename="dcos_results.pkl"):
        """Save DcOS fractal analysis outputs using pickle."""
        data = {
            "results": results,
            "ranges": ranges,
            "delta_break": delta_break,
            "f_break": f_break,
            "delta_upper": delta_upper,
        }
        out_path = Path(self.dfPath or ".") / filename
        with open(out_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Results saved to {out_path}")
        return out_path


    def load_results(self, filename="dcos_results.pkl"):
        """Load DcOS fractal analysis outputs from pickle file."""
        import pickle
        path = Path(self.dfPath or ".") / filename
        if not path.exists():
            raise FileNotFoundError(f"No pickle file found at {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        print(f"Results loaded from {path}")
        return (
            data["results"],
            data["ranges"],
            data["delta_break"],
            data["f_break"],
            data["delta_upper"],
        )
