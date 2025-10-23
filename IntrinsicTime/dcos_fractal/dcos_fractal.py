import pandas as pd
import numpy as np
from scipy.stats import linregress
import os
from pathlib import Path

from dcos_core.dcos_core import DcOS, Sample
from dcos_plot.dcos_plot import DcOS_plotter

class DcOS_fractal:
    def __init__(self,
                 thresholds=None,
                 threshWinLen=7,
                 r2min=0.98,
                 initialMode=-1,
                 debugMode=False):
        if thresholds is None:
            thresholds = np.logspace(-5, -1, 30)  # 0.00001 → 0.1
        self.thresholds = thresholds
        self.threshWinLen = threshWinLen
        self.r2min = r2min
        self.initialMode = initialMode
        self.debugMode = debugMode
        self.df = None
        self.dfPath = None

    # ---------- Utility: data validation ----------
    @staticmethod
    def _validate_input(df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        expected_cols = {"Timestamp", "Price"}
        if not expected_cols.issubset(df.columns):
            raise ValueError(f"Input DataFrame must contain columns {expected_cols}.")
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        if not np.issubdtype(df["Price"].dtype, np.number):
            raise TypeError("Column 'Price' must be numeric.")
        return True

    # ---------- Main computations ----------
    def run_dcos_counts(self, df, thresholds=None, initialMode=-1):
        self._validate_input(df)
        if thresholds is None:
            thresholds = np.logspace(-4, -2, 25)

        results = []
        for delta in thresholds:
            dcos = DcOS(threshold=delta, initialMode=initialMode, midpriceMode=False)
            for _, row in df.iterrows():
                sample = Sample(row["Price"], row["Timestamp"])
                dcos.run(sample)
            results.append((delta, dcos.nDC, dcos.nOS, dcos.nDC + dcos.nOS))

        return pd.DataFrame(results, columns=["threshold", "nDC", "nOS", "nTotal"])

    def fractal_ranges(self, thresholds, freqs):
        delta, f = np.array(thresholds), np.array(freqs)
        mask = (f > 0)
        delta, f = delta[mask], f[mask]
        x, y = np.log10(delta), np.log10(f)

        ranges = []
        for i in range(len(x) - self.threshWinLen + 1):
            xi, yi = x[i:i+self.threshWinLen], y[i:i+self.threshWinLen]
            if not np.all(np.diff(yi) < 0):
                continue
            slope, intercept, r, _, _ = linregress(xi, yi)
            if r**2 >= self.r2min:
                ranges.append((10**x[i], 10**x[i+self.threshWinLen-1], slope, r**2))

        return pd.DataFrame(ranges, columns=["δ_L", "δ_U", "slope", "R2"])

    # ---------- Run pipeline ----------
    def run(self, df=None, dfPath=None, dfName=None, makePlots=True, savePlots=True):
        try:
            if df is None:
                if dfName:
                    if dfName.endswith(".csv"):
                        df = pd.read_csv(os.path.join(dfPath or "", dfName))
                    elif dfName.endswith(".parquet"):
                        df = pd.read_parquet(os.path.join(dfPath or "", dfName))
                    else:
                        raise ValueError("Unsupported input format: must be .csv or .parquet")
                else:
                    raise ValueError("Either df or dfName must be provided.")
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found at path: {dfPath or ''}/{dfName}")

        self.df = df
        self.dfPath = dfPath or os.getcwd()

        results = self.run_dcos_counts(df, thresholds=self.thresholds, initialMode=self.initialMode)
        results["freq"] = results["nTotal"] / len(df)
        results["stderr"] = np.sqrt(results["freq"] * (1 - np.minimum(results["freq"], 1)) / len(df))

        ranges = self.fractal_ranges(results["threshold"], results["freq"])

        if self.debugMode:
            print("Detected fractal ranges:")
            print(ranges)

        if makePlots:
            plt = DcOS_plotter()
            plt.fractal_plot(results, ranges, savePlots=savePlots)

        return results, ranges
