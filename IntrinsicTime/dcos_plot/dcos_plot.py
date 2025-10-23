import plotly.graph_objects as go
import numpy as np
import pandas as pd
from pathlib import Path

class DcOS_plotter:
    def __init__(self, dfPath="."):
        self.dfPath = Path(dfPath)

    def fractal_plot(self, results, ranges, delta_break=None, f_break=None, savePlots=True):
        fig = go.Figure()

        # DC Frequency
        for key, color in [("nEVtot", "#2878d1"), ("nDCtot", "#42b7b0"), ("nOStot", "#b3466c")]:
            freq = results[f"{key}_freq"]
            stderr = results[f"{key}_stderr"]
            fig.add_trace(go.Scatter(
                x=results["threshold"], y=freq,
                mode="lines+markers", name=f"{key} Frequency",
                line=dict(color=color)
            ))
            fig.add_trace(go.Scatter(
                x=np.concatenate([results["threshold"], results["threshold"][::-1]]),
                y=np.concatenate([freq + stderr, (freq - stderr)[::-1]]),
                fill="toself", fillcolor=color.replace("royal", "light"),
                opacity=0.25, line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip", showlegend=False
            ))

        # # Annotate fractal ranges
        # for _, r in ranges.iterrows():
        #     fig.add_vrect(
        #         x0=r["δ_L"], x1=r["δ_U"],
        #         fillcolor="LightGreen", opacity=0.3,
        #         annotation_text=f"slope={r['slope']:.2f},R²={r['R2']:.2f}",
        #         annotation_position="top left"
        #     )

        # Break point annotation
        if np.isfinite(delta_break) and np.isfinite(f_break):
            fig.add_vline(x=delta_break, line_color="black", line_dash="dash")
            fig.add_hline(y=f_break, line_color="black", line_dash="dash")
            fig.add_annotation(
                x=delta_break, y=f_break,
                text=f"δ*: {delta_break:.2e}<br>f*: {f_break:.2e}",
                showarrow=True, arrowhead=1
            )

        fig.update_layout(
            xaxis=dict(title="Threshold δ (log scale)", type="log", autorange=True,
                        titlefont=dict(size=18), tickfont=dict(size=14)),
            yaxis=dict(title="Event Frequency", type="log", range=[-6, 0],
                       titlefont=dict(size=18), tickfont=dict(size=14)),
            title="DcOS Fractal Scaling — DC & OS Frequencies with Error Bands",
            legend=dict(x=0.02, y=0.98, font=dict(size=12)),
            template="plotly_white"
        )

        if savePlots:
            out_path = self.dfPath / "fractal_scaling_plot.html"
            fig.write_html(out_path)
            print(f"Plot saved at {out_path}")

        return fig
