import pandas as pd
import numpy as np
from scipy.stats import linregress
import os
from pathlib import Path
import pickle
import multiprocessing as mp

from IntrinsicTime.dcos_core.dcos_core import DcOS, Sample


class DcOS_fractal:
    """
    Intrinsic Time fractal scaling analysis.
    Now uses safe parallelization (workers receive NumPy arrays, not DataFrames).
    """

    def __init__(self, thresholds=None, initialMode=0, debugMode=False):
        if thresholds is None:
            thresholds = np.logspace(-5, 0, 50)
        self.thresholds = thresholds
        self.initialMode = initialMode
        self.debugMode = debugMode

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

    # ------------------------------ Worker: one threshold ------------------------------
    @staticmethod
    def _run_threshold_full(args):
        δ, arr, initialMode = args
        dcos = DcOS(threshold=δ, initialMode=initialMode, midpriceMode=False)
        seq = []
        # iterate over pre-packed numpy array: columns [Timestamp, Price]
        for timestamp, price in arr:
            code = dcos.run(Sample(price, timestamp))
            if code != 0:
                seq.append(code)
        η = np.log1p(δ)
        norm_os = [seg / η for seg in dcos.osSegment if seg > 0]
        return (
            δ,
            dcos.nDCtot,
            dcos.nOStot,
            dcos.nDCtot + dcos.nOStot,
            np.array(seq, int),
            np.array(norm_os, float),
        )

    # ------------------------------ Parallel runner ------------------------------
    def run_dcos_all_parallel(
        self, df, thresholds=None, initialMode=0, max_workers=None, record_events=True
    ):
        """
        Runs DcOS once per threshold in parallel.
        Uses a NumPy array instead of DataFrame for interprocess safety.
        Returns:
            results (DataFrame),
            event_sequences (dict),
            os_segments (dict)
        """
        self._validate_input(df)
        if thresholds is None:
            thresholds = self.thresholds
        if max_workers is None:
            max_workers = os.cpu_count()

        # convert to lightweight array (Timestamp, Price)
        arr = df[["Timestamp", "Price"]].to_numpy(dtype=float)

        args = [(δ, arr, initialMode) for δ in thresholds]

        with mp.get_context("spawn").Pool(processes=max_workers) as pool:
            outputs = pool.map(self._run_threshold_full, args)

        records = []
        event_sequences, os_segments = {}, {}
        for δ, nDC, nOS, nEV, seq, norm_os in outputs:
            records.append((δ, nDC, nOS, nEV))
            if record_events:
                event_sequences[δ] = seq
                os_segments[δ] = norm_os

        results = pd.DataFrame(records, columns=["threshold", "nDCtot", "nOStot", "nEVtot"])
        return results, event_sequences, os_segments

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

    # ------------------------------ Fit Region ------------------------------
    def determine_fit_region(self, results, pt_constant=61.21, pt_tolerance=2.5):
        dc = results["dc_pct"].values
        δ = results["threshold"].values
        n = len(dc)
        if n == 0:
            results.attrs["δ_min_fit"] = np.nan
            results.attrs["δ_max_fit"] = np.nan
            return results

        if dc[0] < pt_constant:
            idx_start_flat = np.argmax(dc > pt_constant) if np.any(dc > pt_constant) else None
        else:
            idx_start_flat = np.argmax(dc < pt_constant) if np.any(dc < pt_constant) else None
        if idx_start_flat is None or idx_start_flat == 0:
            results.attrs["δ_min_fit"] = np.nan
            results.attrs["δ_max_fit"] = np.nan
            return results

        δ_min_fit = δ[idx_start_flat]

        lo, hi = pt_constant - pt_tolerance, pt_constant + pt_tolerance
        δ_max_fit = np.nan
        idx_end_flat = idx_start_flat
        for j in range(idx_start_flat + 1, n):
            if not (lo < dc[j] < hi):
                idx_end_flat = j - 1
                δ_max_fit = δ[idx_end_flat]
                break
        if not np.isfinite(δ_max_fit):
            idx_end_flat = n - 1
            δ_max_fit = δ[idx_end_flat]

        results.attrs.update(
            {"idx_start_flat": idx_start_flat, "idx_end_flat": idx_end_flat, "δ_min_fit": δ_min_fit, "δ_max_fit": δ_max_fit}
        )
        return results

    # ------------------------------ Tail Fitting ------------------------------
    def analyze_tail_scaling(self, results, δ_min_fit=None, δ_max_fit=None):
        if δ_min_fit is None:
            δ_min_fit = results.attrs.get("δ_min_fit", np.nan)
        if δ_max_fit is None:
            δ_max_fit = results.attrs.get("δ_max_fit", np.nan)

        idx_start = results.attrs.get("idx_start_flat")
        idx_end = results.attrs.get("idx_end_flat")

        # Validate region
        if not np.isfinite(δ_min_fit) or not np.isfinite(δ_max_fit):
            if self.debugMode:
                print("Invalid fit region → skipping regression.")
            return results

        # Extract scaling-zone rows
        mask = (results["threshold"] >= δ_min_fit) & (results["threshold"] <= δ_max_fit)
        trimmed = results.loc[mask].copy()

        if len(trimmed) < 2:
            if self.debugMode:
                print("Not enough points for regression fit.")
            return results

        # ---- Mean & Std %DC over scaling zone ----
        mean_dc_pct = trimmed["dc_pct"].mean()
        std_dc_pct = trimmed["dc_pct"].std()

        # ---- Regression fits ----
        fits = {}
        for key in ["nEVtot_freq", "nDCtot_freq", "nOStot_freq"]:
            mask_valid = trimmed[key] > 0
            x = np.log10(trimmed.loc[mask_valid, "threshold"].values)
            y = np.log10(trimmed.loc[mask_valid, key].values)

            if len(np.unique(x)) < 2:
                fits[key] = {
                    "slope": np.nan,
                    "intercept": np.nan,
                    "r2": np.nan,
                    "pvalue": np.nan
                }
                results[f"y_pred_{key}"] = np.nan
                continue

            slope, intercept, r, pvalue, _ = linregress(x, y)

            fits[key] = {
                "slope": slope,
                "intercept": intercept,
                "r2": r**2,
                "pvalue": pvalue
            }

            results[f"y_pred_{key}"] = 10 ** (intercept + slope * np.log10(results["threshold"]))

        # ---- Store all outputs ----
        results.attrs["tail_fit"] = fits
        results.attrs["fit_region"] = {
            "δ_min_fit": δ_min_fit,
            "δ_max_fit": δ_max_fit
        }

        results.attrs["scaling_zone"] = {
            "idx_min": idx_start,
            "idx_max": idx_end,
            "δ_min": δ_min_fit,
            "δ_max": δ_max_fit,
            "Δ_min": np.log10(δ_min_fit),
            "Δ_max": np.log10(δ_max_fit)
        }

        results.attrs["dc_pct_stats"] = {
            "mean": mean_dc_pct,
            "std": std_dc_pct
        }

        if self.debugMode:
            print("Scaling zone:", results.attrs["scaling_zone"])
            print("Mean %DC:", mean_dc_pct)
            print("Std %DC:", std_dc_pct)
            for k, v in fits.items():
                print(f"{k}: β={v['slope']:.3f}, R²={v['r2']:.3f}, p={v['pvalue']:.3g}")

        return results

    # ------------------------------ Main Pipeline ------------------------------
    def run_count(self, df=None, dfPath=None, dfName=None, parallel=True, record_events=False):
        if df is None:
            if not dfName:
                raise ValueError("Provide either a DataFrame or dfName.")
            ext = Path(dfName).suffix.lower()
            full_path = Path(dfPath or ".") / dfName
            df = pd.read_csv(full_path) if ext == ".csv" else pd.read_parquet(full_path)

        if parallel:
            results, event_sequences, os_segments = self.run_dcos_all_parallel(
                df,
                thresholds=self.thresholds,
                initialMode=self.initialMode,
                record_events=record_events,
            )
        else:
            results, event_sequences, os_segments = self.run_dcos_all_parallel(
                df,
                thresholds=self.thresholds,
                initialMode=self.initialMode,
                max_workers=1,
                record_events=record_events,
            )

        results = self.compute_freqs(results, len(df))
        results.attrs.update(
            {"thresholds": self.thresholds, "event_sequences": event_sequences, "os_segments": os_segments}
        )
        return results

    def run_analysis(self, results, pt_constant=61.21, pt_tolerance=2.5):
        results = self.determine_fit_region(results, pt_constant, pt_tolerance)
        results = self.analyze_tail_scaling(results)
        return results

    def run_count_and_analysis(
        self, df=None, dfPath=None, dfName=None, pt_constant=61.21, pt_tolerance=2.5, parallel=True
    ):
        if df is None:
            if not dfName:
                raise ValueError("Provide either a DataFrame or dfName.")
            ext = Path(dfName).suffix.lower()
            full_path = Path(dfPath or ".") / dfName
            df = pd.read_csv(full_path) if ext == ".csv" else pd.read_parquet(full_path)

        results = self.run_count(df, dfPath, dfName, parallel)
        results = self.run_analysis(results, pt_constant, pt_tolerance)
        return results

    def run(
        self,
        df=None,
        dfPath=None,
        dfName=None,
        pt_constant=61.21,
        pt_tolerance=2.5,
        parallel=True,
        record_events=False,
    ):
        """
        High-level convenience pipeline.

        Parameters
        ----------
        df : pd.DataFrame, optional
            DataFrame with columns ['Timestamp', 'Price'].
        dfPath : str or Path, optional
            Directory where dfName is located (if df is not given).
        dfName : str, optional
            CSV or Parquet file name (if df is not given).
        pt_constant : float, default 61.21
            Target %DC used to identify the scaling region.
        pt_tolerance : float, default 2.5
            Allowed deviation around pt_constant in %.
        parallel : bool, default True
            Whether to run thresholds in parallel.
        record_events : bool, default False
            Whether to keep full event sequences and OS segments.

        Returns
        -------
        results : pd.DataFrame
            Per-threshold counts, frequencies and regression output.
        fit_region : dict
            Dictionary with δ_min_fit and δ_max_fit (may be NaN).
        """
        results = self.run_count(
            df=df,
            dfPath=dfPath,
            dfName=dfName,
            parallel=parallel,
            record_events=record_events,
        )
        results = self.run_analysis(
            results,
            pt_constant=pt_constant,
            pt_tolerance=pt_tolerance,
        )
        fit_region = results.attrs.get(
            "fit_region",
            {
                "δ_min_fit": results.attrs.get("δ_min_fit", np.nan),
                "δ_max_fit": results.attrs.get("δ_max_fit", np.nan),
            },
        )
        return results, fit_region

    # ------------------------------ Save / Load ------------------------------
    def save_results(self, results, dfPath=None, dfName="dcos_results"):
        """
        Saves results as:
            <dfName>_results.csv  – DataFrame
            <dfName>_attrs.csv    – attributes dictionary (flattened)
        """
        pkl_path = Path(dfPath or ".") / f"{dfName}.pkl"
        bundle = {
            "results": results,
            "attrs": results.attrs
        }
        with open(pkl_path, "wb") as f:
            pickle.dump(bundle, f)
        if self.debugMode:
            print(f"Saved results to {pkl_path}")

        # ---------------------- Save DataFrame ----------------------
        df_path = Path(dfPath or ".") / f"{dfName}.csv"
        results.to_csv(df_path, index=False)

        # ---------------------- Save Attributes ----------------------
        attrs_path = Path(dfPath or ".") / f"{dfName}_attrs.csv"

        # Flatten attributes: convert non-scalars (dicts etc.) into repr strings
        flat_attrs = {}
        for k, v in results.attrs.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                flat_attrs[k] = v
            else:
                # store dicts, lists, arrays in readable JSON-style
                try:
                    flat_attrs[k] = repr(v)
                except Exception:
                    flat_attrs[k] = str(v)

        # Save as CSV with columns: key, value
        pd.DataFrame(
            [{"key": k, "value": flat_attrs[k]} for k in flat_attrs]
        ).to_csv(attrs_path, index=False)
        if self.debugMode:
            print(f"Saved DataFrame to   {df_path}")
            print(f"Saved attributes to  {attrs_path}")
        return self


    def load_results(self, dfPath=None, dfName="dcos_results.pkl"):
        path = Path(dfPath or ".") / dfName
        if not path.exists():
            raise FileNotFoundError(f"No pickle file found at {path}")
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        results = bundle["results"]
        for k, v in bundle["attrs"].items():
            results.attrs[k] = v
        if self.debugMode:
            print(f"Loaded results from {path}")
        return results
