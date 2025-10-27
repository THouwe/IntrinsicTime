import plotly.graph_objects as go
import numpy as np
import pandas as pd
from pathlib import Path

class DcOS_plotter:
    def __init__(self, dfPath="."):
        self.dfPath = Path(dfPath)

    def fractal_plot(self, results, ranges, delta_break=None, f_break=None, savePlots=True):
        fig = go.Figure()

        # compute derived columns
        results["dc_pct"] = 100 * results["nDCtot_freq"] / results["nEVtot_freq"]
        p = results["nDCtot_freq"] / results["nEVtot_freq"]
        n = results["nEVtot"]
        results["dc_pct_stderr"] = 100 * np.sqrt(p * (1 - p) / np.maximum(n, 1))

        # trim valid range
        valid_mask = (results["dc_pct"] > 0) & (results["dc_pct"] < 100)
        last_valid = results.attrs.get("last_valid_idx", valid_mask[::-1].idxmax())
        trimmed = results.loc[:last_valid].copy()

        # main frequency curves
        for key, color in [("nEVtot", "#2878d1"), ("nDCtot", "#42b7b0"), ("nOStot", "#b3466c")]:
            freq = trimmed[f"{key}_freq"]
            stderr = trimmed[f"{key}_stderr"]
            fig.add_trace(go.Scatter(
                x=trimmed["threshold"], y=freq,
                mode="lines+markers", name=f"{key} Frequency",
                line=dict(color=color, width=2), opacity=0.7, yaxis="y1"
            ))
            fig.add_trace(go.Scatter(
                x=np.concatenate([trimmed["threshold"], trimmed["threshold"][::-1]]),
                y=np.concatenate([freq + stderr, (freq - stderr)[::-1]]),
                fill="toself", fillcolor=color, opacity=0.15,
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip", showlegend=False, yaxis="y1"
            ))

        # fitted lines extended until f=1
        fits = results.attrs.get("tail_fit", {})
        if fits:
            full_x = results["threshold"].values
            log_x = np.log10(full_x)
            for key, color in [("nEVtot_freq", "#2878d1"), ("nDCtot_freq", "#42b7b0"), ("nOStot_freq", "#b3466c")]:
                if key in fits:
                    slope, intercept = fits[key]["slope"], fits[key]["intercept"]
                    y_fit = 10 ** (intercept + slope * log_x)
                    mask = y_fit <= 1
                    x_cut, y_cut = full_x[mask], y_fit[mask]

                    fig.add_trace(go.Scatter(
                        x=x_cut, y=y_cut,
                        mode="lines",
                        line=dict(color=color, dash="dash", width=1.5),
                        name=f"{key.replace('_freq','')} tail fit (β={slope:.2f})",
                        opacity=0.8, yaxis="y1"
                    ))

        # add vertical line at δ_upper_cutoff if present
        δ_upper = results.attrs.get("delta_upper_cutoff", None)
        if δ_upper and np.isfinite(δ_upper):
            fig.add_vline(
                x=δ_upper,
                line=dict(color="gray", dash="dot", width=1),
                annotation_text=f"δ_upper={δ_upper:.2e}",
                annotation_position="top right"
            )

        # secondary axis: % DC / total
        fig.add_trace(go.Scatter(
            x=trimmed["threshold"], y=trimmed["dc_pct"],
            mode="lines+markers", name="% DC / Total",
            line=dict(color="black", dash="dot", width=1.5),
            opacity=0.6, yaxis="y2"
        ))
        fig.add_trace(go.Scatter(
            x=np.concatenate([trimmed["threshold"], trimmed["threshold"][::-1]]),
            y=np.concatenate([
                trimmed["dc_pct"] + trimmed["dc_pct_stderr"],
                (trimmed["dc_pct"] - trimmed["dc_pct_stderr"])[::-1]
            ]),
            fill="toself", fillcolor="rgba(0,0,0,0.1)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip", showlegend=False, yaxis="y2"
        ))

        # layout
        fig.update_layout(
            xaxis=dict(title="Threshold δ (log scale)", type="log"),
            yaxis=dict(title="Event Frequency", type="log", range=[-6, 0],
                       titlefont=dict(size=18, color="#2878d1"),
                       tickfont=dict(size=14, color="#2878d1")),
            yaxis2=dict(title="% DC over total", overlaying="y", side="right", type="linear",
                        titlefont=dict(size=16, color="black"),
                        tickfont=dict(size=12, color="black")),
            title="DcOS Fractal Scaling — Tail Fits, δ_upper_cutoff, and %DC/Total",
            legend=dict(x=0.02, y=0.98, font=dict(size=12)),
            template="plotly_white"
        )

        if savePlots:
            full_path = self.dfPath / "fractal_scaling_full.html"
            fig.write_html(full_path)
            print(f"Full plot saved at {full_path}")

            # cropped version up to δ_upper_cutoff
            if δ_upper and np.isfinite(δ_upper):
                cropped_path = self.dfPath / "fractal_scaling_cropped.html"
                xaxis_range = [min(trimmed["threshold"]), δ_upper]
                fig_cropped = fig.update_layout(xaxis=dict(range=xaxis_range, type="log"))
                fig_cropped.write_html(cropped_path)
                print(f"Cropped plot saved at {cropped_path}")

        return fig
