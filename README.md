# IntrinsicTime

Utilities for decomposing **intrinsic time** events and analyzing **fractal scaling** behavior in price or signal data.

This package provides:
- `dcos_core`: A **Directional Change and Overshoot (DcOS)** event detector.
- `dcos_fractal`: Tools for **fractal scaling** and **multi-threshold analysis**.
- `dcos_plot`: **Plotly** visualization for interactive fractal plots.

---

## Installation

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

IntrinsicTime decomposes time series into Directional Changes (DCs) and Overshoots (OSs) based on log-scale thresholds.
It then explores the fractal scaling law between event frequency and detection threshold.

### Example Usage
```
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
