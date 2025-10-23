## DcOS events are first‑passage moves of size δ in log price
Robust power‑law behaviour can be observed for many phenomena.
For instance, Intrinsic Time event density of BTCUSDT price ticks scale linearly with DcOS δ threshold in log space (cit), consistently with first‑passage theory (cit) plus market microstructure.
However, this is the case only within a given range of δ thresholds, as the power law may brake at 'extremely low' or 'extremely high' δs.

For small δs, issues relate to **microstructure noise** (tick size, latency, and irregular sampling inject high‑frequency mean reversion. This raises event frequency toward a ceiling and flattens the log–log curve) and **discretization limits** (time and sample - e.g., *price* - granularity cap how many distinct first‑passage events you can observe).

For large δs, issues relate to **data scarcity**: too few events reduce fit quality and increase variance.

## Fractal brakepoint formalization
Compute local slopes (b(\delta)) with a sliding window (w) on ((\log \delta,\log f)).
Mark the smallest δ where either (R^2 < R^2_{\min}) or (|\Delta b|) exceeds one standard error from adjacent windows.
Your windowed method will return something close to δ ≈ 6e‑4 for this dataset if your visual read is correct.
