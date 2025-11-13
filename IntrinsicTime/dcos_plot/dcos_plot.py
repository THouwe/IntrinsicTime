import plotly.graph_objects as go
import numpy as np
import pandas as pd
from pathlib import Path
import os


class DcOS_plotter:
    def __init__(self, dfPath="."):
        self.dfPath = Path(dfPath)


    def fractal_plot(self, results, pt_constant=61.21, pt_tolerance=2.5,
                     savePlots=True, filename="dcos_fractal.html", dfPath=None):
        fig = go.Figure()

        # --- recompute DC % and errors ---
        results["dc_pct"] = 100 * results["nDCtot_freq"] / results["nEVtot_freq"]
        p = results["nDCtot_freq"] / results["nEVtot_freq"]
        n = results["nEVtot"]
        results["dc_pct_stderr"] = 100 * np.sqrt(p * (1 - p) / np.maximum(n, 1))

        # --- remove highest δ point ---
        results = results.iloc[:-1, :].copy()

        # --- region boundaries from attributes ---
        δ_min_fit = results.attrs.get("δ_min_fit", np.nan)
        δ_max_fit = results.attrs.get("δ_max_fit", np.nan)
        tail_fit = results.attrs.get("tail_fit", {})

        # --- main frequency curves ---
        for key, color in [("nEVtot", "#2878d1"),
                           ("nDCtot", "#42b7b0"),
                           ("nOStot", "#b3466c")]:
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

        # --- highlight ±pt_tolerance% DC region ---
        if np.isfinite(δ_min_fit) and np.isfinite(δ_max_fit):
            fig.add_vrect(
                x0=δ_min_fit, x1=δ_max_fit,
                fillcolor="yellow", opacity=0.25, layer="below", line_width=0,
                annotation_text=f"{pt_constant} ± {pt_tolerance}% DC region",
                annotation_position="top left"
            )

        # --- regression fits (safe handling) ---
        for key, color in [("nEVtot_freq", "#2878d1"),
                           ("nDCtot_freq", "#42b7b0"),
                           ("nOStot_freq", "#b3466c")]:
            col = f"y_pred_{key}"
            if col not in results.columns:
                continue
            try:
                slope = tail_fit.get(key, {}).get("slope", np.nan)
                fig.add_trace(go.Scatter(
                    x=results["threshold"], y=results[col],
                    mode="lines",
                    name=f"{key.replace('_freq', '')} final fit (β={slope:.2f})",
                    line=dict(color=color, dash="dot", width=1.6),
                    opacity=0.9, yaxis="y1"
                ))
            except Exception:
                continue  # skip invalid fit safely

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

        # --- crop x-axis to last valid δ ---
        mask_valid = (results["nDCtot"] >= 1) & (results["nOStot"] >= 1)
        δ_crop = results.loc[mask_valid, "threshold"].iloc[-1] if np.any(mask_valid) else max(results["threshold"])

        # --- layout ---
        fig.update_layout(
            xaxis=dict(
                title="Threshold δ (log scale)",
                type="log",
                range=[np.log10(min(results["threshold"])), np.log10(δ_crop)]
            ),
            yaxis=dict(
                title=dict(
                    text="Event Frequency",
                    font=dict(size=18, color="#2878d1")
                ),
                type="log",
                range=[-6, 0],
                tickfont=dict(size=14, color="#2878d1"),
                gridcolor="#2878d1"
            ),
            yaxis2=dict(
                title=dict(
                    text="% DC over total",
                    font=dict(size=16, color="black")
                ),
                overlaying="y",
                side="right",
                type="linear",
                tickfont=dict(size=12, color="black")
            ),
            title=f"DcOS Fractal Scaling — Final {pt_constant} ± {pt_tolerance}% Region Fit and %DC/Total",
            legend=dict(x=0.02, y=0.98, font=dict(size=12)),
            template="plotly_white"
        )

        # --- save plot ---
        if savePlots:
            save_dir = Path(dfPath or self.dfPath)
            save_dir.mkdir(parents=True, exist_ok=True)
            full_path = save_dir / filename
            fig.write_html(full_path)
            print(f"Full-range plot saved at {full_path}")

        return fig
