import plotly.graph_objects as go
import numpy as np
import pandas as pd
from pathlib import Path
import os


class DcOS_plotter:
    def __init__(self, dfPath="."):
        self.dfPath = Path(dfPath)

    def fractal_plot(self, results, low_pt=60, high_pt=70, savePlots=True, filename="dcos_fractal.html", dfPath=None):
        fig = go.Figure()

        # recompute DC % and errors
        results["dc_pct"] = 100 * results["nDCtot_freq"] / results["nEVtot_freq"]
        p = results["nDCtot_freq"] / results["nEVtot_freq"]
        n = results["nEVtot"]
        results["dc_pct_stderr"] = 100 * np.sqrt(p * (1 - p) / np.maximum(n, 1))

        # --- get region boundaries ---
        δ_min_fit = results.attrs.get("δ_min_fit", np.nan)
        δ_max_fit = results.attrs.get("δ_max_fit", np.nan)
        tail_fit = results.attrs.get("tail_fit", {})

        # also find δ_high (first DC ≥ high%)
        mask_high = results["dc_ratio"] >= high_pt / 100
        δ_high = results.loc[mask_high, "threshold"].iloc[0] if np.any(mask_high) else np.nan

        # --- main frequency curves ---
        for key, color in [("nEVtot", "#2878d1"), ("nDCtot", "#42b7b0"), ("nOStot", "#b3466c")]:
            freq = results[f"{key}_freq"]
            stderr = results[f"{key}_stderr"]
            fig.add_trace(go.Scatter(
                x=results["threshold"], y=freq,
                mode="lines+markers", name=f"{key} Frequency",
                line=dict(color=color, width=2), opacity=0.8, yaxis="y1"
            ))
            fig.add_trace(go.Scatter(
                x=np.concatenate([results["threshold"], results["threshold"][::-1]]),
                y=np.concatenate([freq + stderr, (freq - stderr)[::-1]]),
                fill="toself", fillcolor=color, opacity=0.15,
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip", showlegend=False, yaxis="y1"
            ))

        # --- highlight low–high% DC region ---
        if np.isfinite(δ_min_fit) and np.isfinite(δ_high):
            fig.add_vrect(
                x0=δ_min_fit, x1=δ_high,
                fillcolor="yellow", opacity=0.25, layer="below", line_width=0,
                annotation_text=f"{low_pt}–{high_pt}% DC region", annotation_position="top left"
            )

        # --- add regression fits (dotted) ---
        for key, color in [("nEVtot_freq", "#2878d1"),
                           ("nDCtot_freq", "#42b7b0"),
                           ("nOStot_freq", "#b3466c")]:
            if f"y_pred_{key}" in results.columns:
                fig.add_trace(go.Scatter(
                    x=results["threshold"], y=results[f"y_pred_{key}"],
                    mode="lines", name=f"{key.replace('_freq', '')} final fit (β={tail_fit.get(key, {}).get('slope', np.nan):.2f})",
                    line=dict(color=color, dash="dot", width=1.6),
                    opacity=0.9, yaxis="y1"
                ))

        # --- secondary axis: % DC / total ---
        fig.add_trace(go.Scatter(
            x=results["threshold"], y=results["dc_pct"],
            mode="lines+markers", name="% DC / Total",
            line=dict(color="black", dash="dot", width=1.5),
            opacity=0.7, yaxis="y2"
        ))
        fig.add_trace(go.Scatter(
            x=np.concatenate([results["threshold"], results["threshold"][::-1]]),
            y=np.concatenate([
                results["dc_pct"] + results["dc_pct_stderr"],
                (results["dc_pct"] - results["dc_pct_stderr"])[::-1]
            ]),
            fill="toself", fillcolor="rgba(0,0,0,0.1)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip", showlegend=False, yaxis="y2"
        ))

        # --- crop x-axis to largest delta with ≥1 event per type ---
        mask_valid = (results["nDCtot"] >= 1) & (results["nOStot"] >= 1)
        δ_crop = results.loc[mask_valid, "threshold"].iloc[-1] if np.any(mask_valid) else max(results["threshold"])

        # --- layout ---
        fig.update_layout(
            xaxis=dict(title="Threshold δ (log scale)", type="log",
                       range=[np.log10(min(results["threshold"])), np.log10(δ_crop)]),
            yaxis=dict(title="Event Frequency", type="log", range=[-6, 0],
                       titlefont=dict(size=18, color="#2878d1"),
                       tickfont=dict(size=14, color="#2878d1")),
            yaxis2=dict(title="% DC over total", overlaying="y", side="right", type="linear",
                        titlefont=dict(size=16, color="black"),
                        tickfont=dict(size=12, color="black")),
            title=f"DcOS Fractal Scaling — Final {low_pt}–{high_pt}% Region Fit and %DC/Total",
            legend=dict(x=0.02, y=0.98, font=dict(size=12)),
            template="plotly_white"
        )

        # --- save plots ---
        if savePlots:
            # full version
            if dfPath is None:
                full_path = os.path.join(self.dfPath, filename)
            else:
                full_path = os.path.join(dfPath, filename)
            fig.write_html(full_path)
            print(f"Full-range plot saved at {full_path}")

            # # cropped version for 60–70% region only
            # if np.isfinite(δ_min_fit) and np.isfinite(δ_high):
            #     cropped_path = self.dfPath / "fractal_scaling_finalfit_zoomed.html"
            #     fig_zoomed = fig.update_layout(
            #         xaxis=dict(range=[np.log10(δ_min_fit), np.log10(δ_high)], type="log")
            #     )
            #     fig_zoomed.write_html(cropped_path)
            #     print(f"Cropped region plot saved at {cropped_path}")

        return fig
