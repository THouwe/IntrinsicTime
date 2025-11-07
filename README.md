# IntrinsicTime

Utilities for decomposing **intrinsic time** events and analyzing **fractal scaling** behavior in price or signal data.

This package provides:
- [dcos_core](dcos_core.md): A **Directional Change and Overshoot (DcOS)** event detector.
- [dcos_fractal](dcos_fractal.md): Tools for **fractal scaling** and **multi-threshold analysis**.
- [dcos_plot](dcos_plot.md): **Plotly** visualization for interactive fractal plots.

---

## Installation

### From PyPi
```
pip install IntrinsicTime
```

### From GitHub
```
pip install git+https://github.com/THouwe/IntrinsicTime.git
```

### Local Install
```
git clone https://github.com/THouwe/IntrinsicTime.git
cd IntrinsicTime
pip install -e .
```

### Dependencies
See `requirements.txt`.

---

## Overview

## DcOS events are first‑passage moves of size δ in log space
Robust power‑law behaviour can be observed for many phenomena.
For instance, Intrinsic Time event density of BTCUSDT price ticks scale linearly with DcOS δ threshold in log space (cit), consistently with first‑passage theory (cit) plus market microstructure.
However, this is the case only within a given range of δ thresholds, as the power law may brake at 'extremely low' or 'extremely high' δs.

For small δs, issues relate to **microstructure noise** (tick size, latency, and irregular sampling inject high‑frequency mean reversion. This raises event frequency toward a ceiling and flattens the log–log curve) and **discretization limits** (time and sample - e.g., *price* - granularity cap how many distinct first‑passage events you can observe).

For large δs, issues relate to **data scarcity**: too few events reduce fit quality and increase variance.

## Fractal brakepoint formalization
Compute local slopes (b(\delta)) with a sliding window (w) on ((\log \delta,\log f)).
Mark the smallest δ where either (R^2 < R^2_{\min}) or (|\Delta b|) exceeds one standard error from adjacent windows.
Your windowed method will return something close to δ ≈ 6e‑4 for this dataset if your visual read is correct.


### Example Usage
```
import numpy as np
import pandas as pd
from IntrinsicTime import DcOS_fractal

# Example input DataFrame
df = pd.DataFrame({
    "Timestamp": range(1000),
    "Price": 100 + np.cumsum(np.random.randn(1000))
})

# Initialize and run
analyzer = DcOS_fractal(debugMode=True)
results, ranges = analyzer.run(df)

# Display results
print(results.head())
print(ranges)
```
