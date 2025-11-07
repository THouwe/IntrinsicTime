# DcOS_core

## Introduction

The Directional-change/OverShoot (**DcOS**) analysis framework has been used to unveil scaling law distributions across a growing number of time series, mostly financial assets, e.g., [FX data](https://www.sg.ethz.ch/publications/2011/glattfelder2011patterns-in-high-frequency/Patterns_in_high_frequency_FX_data_discovery_of_12_empirical_scaling_laws.pdf).
However, the methodology has undergone a few changes throughout its long history.
I have authored this python module to provide a definition of DcOS in the hope that it can be used as a standard going forward.
The aim was to build a DcOS operator that was as simple, universal and unbiased as possible.
Although I will refer to the data / samples as *price*, the method can be applied to any time series.

---

## DcOS Operator

The **DcOS operator** is a state machine that detects regime changes in price movements using *intrinsic time*. It processes price samples sequentially and identifies **directional changes (DCs)** and **overshoots (OSs)** based on a fixed *logarithmic threshold*.

---

## Core Concept

The operator models market dynamics by switching between two *modes*:
- **Up mode (-1):** Price is currently rising.
- **Down mode (+1):** Price is currently falling.

A **directional change (DC)** is triggered when the price reverses by at least the logarithmic threshold  
`η = log(1 + threshold)` relative to the most recent extreme.  
An **overshoot (OS)** is the continuation of movement in the same direction *beyond* the threshold before the next DC.

---

## Data Structures

- **`Sample(level, time)`** — encapsulates a price observation.
- **`Price(bid, ask, time)`** — used when working with bid/ask quotes; can compute midprice via `getMid()`.

---

## Internal State

`DcOS` maintains a memory of the current and previous turning points:

- `mode`: current direction state.
- `extreme`: latest local maximum/minimum.
- `reference`: comparison base for overshoot detection.
- `DC` and `prevDC`: most recent and previous directional change points.
- `osL`: overshoot length (in log space).
- `dcL`: magnitude of latest directional change.
- Counters:
  - `nOSseq`, `nOStot`: overshoot counts (current sequence / total).
  - `nDCseq`, `nDCtot`: directional change counts (current sequence / total).

---

## Algorithm Logic

1. **Initialization**  
   On the first sample, all reference points are set and no event is emitted.

2. **Neutral Mode Detection**  
   Before any trend is established, the operator waits for a price move of size ≥ threshold to define an initial direction (up or down).

3. **Overshoot Phase (Continuation)**  
   If price extends further in the same direction as the current mode:
   - Update `extreme` (new high or low).
   - If the move from `reference` exceeds another threshold:
     - Register an overshoot (`osL`).
     - Increment counters.
     - Return code `2 * side` (+2 for up, −2 for down).

4. **Directional Change Phase (Reversal)**  
   If price moves *against* the current mode by at least one threshold:
   - Record a new directional change.
   - Update `prevExtreme`, `prevDC`, and flip the mode.
   - Reset overshoot counters.
   - Return code `−side` (−1 for up→down, +1 for down→up).

5. **Otherwise**  
   If neither condition is met, return `0` (no event).

---

## Summary

**DcOS** converts a raw time series into a sequence of intrinsic events—directional changes and overshoots—based on a symmetric logarithmic threshold.  
It serves as the core operator for building directional-change-based analytics in *Intrinsic Time* frameworks.
