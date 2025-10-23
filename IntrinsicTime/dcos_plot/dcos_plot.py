import pandas as pd
import numpy as np
import os
from pathlib import Path
import plotly.graph_objects as go


class DcOS_plotter:
    def __init__(self, savePlots=True):
        self.savePlots = savePlots

    def fractal_plot(self, results, ranges, savePlots=True):
        if not isinstance(results, pd.DataFrame):
            raise TypeError("results must be a pandas DataFrame.")

        fig = go.Figure()

        # Left Y-axis: Frequency with error bands
        freq = results["freq"]
        stderr = results["stderr"]

        fig.add_trace(go.Scatter(
            x=results["threshold"], y=freq,
            mode="lines+markers", name="Event Frequency",
            yaxis="y1", line=dict(color="royalblue")
        ))
        fig.add_trace(go.Scatter(
            x=np.concatenate([results["threshold"], results["threshold"][::-1]]),
            y=np.concatenate([freq + stderr, (freq - stderr)[::-1]]),
            fill="toself", fillcolor="lightblue", opacity=0.3,
            line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip",
            showlegend=False, yaxis="y1"
        ))

        # Right Y-axis: Event counts
        fig.add_trace(go.Scatter(
            x=results["threshold"], y=results["nEVtot"],
            mode="lines+markers", name="Event Count",
            yaxis="y2", line=dict(color="firebrick", dash="dot")
        ))

        # Highlight fractal ranges
        for _, r in ranges.iterrows():
            fig.add_vrect(
                x0=r["δ_L"], x1=r["δ_U"],
                fillcolor="LightGreen", opacity=0.3,
                annotation_text=f"slope={r['slope']:.2f}, R²={r['R2']:.2f}",
                annotation_position="top left"
            )

        fig.update_layout(
            xaxis=dict(title="Threshold δ (log scale)", type="log", autorange=True),
            yaxis=dict(title="Event Frequency", type="log", range=[-5, -1],
                       titlefont=dict(size=18), tickfont=dict(size=14)),
            yaxis2=dict(title="Event Count", type="log", range=[1, 5],
                        overlaying="y", side="right",
                        titlefont=dict(size=18), tickfont=dict(size=14)),
            title="DcOS Fractal Scaling with Error Bands",
            legend=dict(x=0.02, y=0.98, font=dict(size=12)),
            template="plotly_white"
        )

        if savePlots:
            out_path = Path(self.dfPath) / "fractal_scaling_plot.html"
            fig.write_html(out_path)
            print(f"Plot saved at {out_path}")

        return fig
