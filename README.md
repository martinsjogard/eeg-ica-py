# ICA Tools for EEG/MEG – Python Implementation

This repository contains a collection of Python functions for Independent Component Analysis (ICA) in EEG/MEG signal processing. These functions assist with estimating ICA components, analyzing their properties, correcting for artifacts, and visualizing results.

The prefix `msfun_` is used to denote modular signal processing functions.

## Overview

These functions support a full ICA pipeline including:

- Estimating ICA from preprocessed data
- Post hoc analyses (e.g., power spectrum, cumulants, correlation with external signals)
- Identifying artifactual components
- Visualizing component properties
- Removing artifactual components from data

---

## Function Index

### `msfun_eeg_ica_estimate.py`
Estimates ICA from EEG data.

- **Inputs**: EEG data array (`data`), configuration dictionary (`cfg`)
- **Outputs**: Dictionary `IC` with keys like `'A'`, `'S'`, and optional diagnostics

---

### `msfun_meg_ica.py`
Estimates ICA from MEG data using a configuration dictionary.

- **Inputs**: MEG data (`data`), configuration (`cfg`)
- **Outputs**: ICA components in dictionary format (`IC`)

---

### `msfun_meg_ica_corranalysis.py`  
Computes correlation between ICs and external signals.

- **Inputs**: ICA structure `IC`, external signal array `sig`, config `cfg`
- **Outputs**: Updated `IC` with `'corranalysis'` field

---

### `msfun_meg_ica_corranalysis2.py`
Computes trial-by-trial correlation between ICs and an external signal.

- **Inputs**: `IC`, external `sig`, and config
- **Outputs**: Updated `IC['corrtrial']` with correlation data

---

### `msfun_meg_ica_cumulantanalysis.py`
Analyzes non-Gaussianity of ICs using 4th-order cumulants.

- **Inputs**: ICA structure
- **Outputs**: Updated `IC['cumulant']` with kurtosis metrics

---

### `msfun_meg_ica_estimate.py`
Performs ICA decomposition on MEG data with multiple config options.

- **Inputs**: MEG signal, config
- **Outputs**: ICA result dictionary `IC`

---

### `msfun_meg_ica_ndof.py`
Computes number of degrees of freedom for IC components.

- **Inputs**: ICA structure
- **Outputs**: Updated `IC['ndof']`

---

### `msfun_meg_ica_powerspectrum.py`
Computes power spectral density for each IC.

- **Inputs**: `IC`, sampling frequency `sfreq`
- **Outputs**: Updated `IC['freq']` and `IC['powspctrm']`

---

### `msfun_meg_ica_resultfig.py`
Generates result summary plots for ICA components.

- **Inputs**: `IC`, optional config
- **Outputs**: Visual output (figures)

---

### `msfun_meg_ica_spectralanalysis.py`
Fits linear or power-law models to IC power spectra and computes goodness-of-fit.

- **Inputs**: `IC`, config
- **Outputs**: Updated `IC['spectral']`

---

### `msfun_meg_ica_viewer.py`
**Placeholder**: Intended for interactive visualization of ICA results.

- **Inputs**: raw data, times, ICA struct, optional config
- **Outputs**: None (GUI functionality to be implemented)

---

### `msfun_remartIC.py`
Removes artifact-related ICs from EEG/MEG data and optionally visualizes before/after.

- **Inputs**: Raw data, ICA struct, list of bad components, plot flag
- **Outputs**: Cleaned data array

---

## Usage Notes

- All functions are written in pure Python using standard packages (`numpy`, `matplotlib`, `scipy`).
- ICA results are stored in Python dictionaries using keys like `'A'`, `'S'`, and additional computed results.
- Many functions produce visual outputs and use `matplotlib` for plotting.
- For full pipelines, begin with estimation (`*_estimate.py`), run analysis functions, and use `msfun_remartIC.py` to remove artifact components.
