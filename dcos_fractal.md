# DcOS_fractal

## Introduction

Fractal analysis is complicated by 2 main factors:
1. it is a computation- and data-hungry method
2. methods to establish whether distributions are truly fractal are lacking
This python module is aimed at overcoming both limitations by providing:
1. a way to run the `DcOS_core` module efficiently and in parallel, dramatically reducing computation time for multi-scale analysis
2. a method to establish whether, and within which bounds, is the observed distribution fractal.
It also provides a number of useful information regarding the structure of the data.

---

## DcOS events are first‑passage moves of size δ in log price
Power‑law behaviour can be observed ubiquitously in the crypto market.
For instance, Intrinsic Time event density of BTCUSDT price ticks scale linearly with DcOS δ threshold in log space, consistently with first‑passage theory plus market microstructure.
However, this is the case only within a given range of δ thresholds, as the power law may brake at 'extremely low' or 'extremely high' δs.

For small δs, issues relate to **microstructure noise** (tick size, latency, and irregular sampling inject high‑frequency mean reversion; this raises event frequency toward a ceiling and flattens the log–log curve) and **discretization limits** (time and sample - e.g., *price* - granularity cap how many distinct first‑passage events you can observe).

For large δs, issues relate to **data scarcity**: too few events reduce fit quality and increase variance.

Although it can be assumed that without these physical and informational constraints, power laws hold true infinitely, analyses that assume scaling-law behaviour need to be aware of the limits within which they are applicable.
The **`DcOS_fractal`** will do this for you.

---

## Logic

The **`DcOS_fractal`** class extends the `DcOS` operator by performing **fractal scaling analysis** on event frequencies across multiple thresholds.  
It estimates how the counts and frequencies (**event densities**) of **directional changes (DCs)** and **overshoots (OSs)** scale as a function of the logarithmic threshold `δ`.

For a range of thresholds `δ`, the class:
1. Runs the **DcOS operator** to count DC and OS events.
2. Computes their **relative frequencies** and **ratios**.
3. Identifies a **fitting region** in the fractal scaling curve (%DC vs. δ).
4. Performs **log–log regressions** to estimate scaling exponents.

### 1. Threshold Run
Can be single- or multi-threaded (default).

### 2. Event frequency Computation
Number of events are collected. Raw counts are converted into normalized frequencies (computes per-tick event frequencies and standard errors).
This step quantifies how often each type of intrinsic event occurs at each threshold.

### 3. Fit Region Identification
Assumes that distribution can be split into 3 regimes: high-frequency-noise (microstructure-dominated), scaling-law, and low-frequency-noise (data scarcity).
Under high-frequency-noise, dc/os ratio is not constant. `low_pt` frequency (default = 61.5) acts as an attractor: data will eventually cross this threshold.
Under scaling-law, dc/os ratio is constant. Regression coefficients of event-types are equal, lines are parallel
Under low-frequency-noise, dc/os ratio is not constant once again. When it moves by +- `high_pt_change` (default = 4%) from `low_pt`, fit is considered broken

### 4. Log–log regression
To estimate scaling exponents within the fit region.
