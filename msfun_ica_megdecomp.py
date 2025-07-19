import numpy as np

def msfun_ica_meg_estimate(raw, data, extdata=None, cfg=None):
    '''
    Applies temporal ICA decomposition to MEG data, optionally including 
    correlation analysis with external signals.

    Parameters:
    - raw: dict or object representing MEG header (from mne.io.read_raw_fif)
    - data: numpy array, shape [N x T] or [K x N x T]
    - extdata: numpy array, shape matching data (optional)
    - cfg: dict with optional keys:
        - 'ica': config for ICA estimation
        - 'cumulant': config for cumulant analysis
        - 'corranalysis': bool
        - 'corr': config for correlation analysis
        - 'spectralanalysis': bool
        - 'fft': config for FFT
        - 'spectral': config for spectral analysis
        - 'viewer': config for ICA viewer

    Returns:
    - IC: dict containing ICA components and analysis results
    '''
    from msfun_sig_concat_epoch import msfun_sig_concat_epoch
    from msfun_meg_ica_estimate import msfun_meg_ica_estimate
    from msfun_meg_ica_cumulantanalysis import msfun_meg_ica_cumulantanalysis
    from msfun_meg_ica_corranalysis import msfun_meg_ica_corranalysis
    from msfun_meg_ica_powerspectrum import msfun_meg_ica_powerspectrum
    from msfun_meg_ica_spectralanalysis import msfun_meg_ica_spectralanalysis
    from msfun_meg_ica_resultfig import msfun_meg_ica_resultfig
    from msfun_meg_ica_viewer import msfun_meg_ica_viewer

    if cfg is None:
        cfg = {}
    if extdata is None:
        extdata = np.array([])

    if not isinstance(raw, dict):
        raise ValueError("raw must be a struct-like header object")

    shape = data.shape
    if len(shape) == 2:
        epoching = False
        N, T = shape
        if extdata.size != 0 and (extdata.ndim != 2 or extdata.shape[1] != T):
            raise ValueError("Inconsistent extdata shape with data")
        S = extdata.shape[0] if extdata.size != 0 else 0
    elif len(shape) == 3:
        epoching = True
        K, N, T = shape
        if extdata.size != 0 and (extdata.ndim != 3 or extdata.shape[0] != K or extdata.shape[2] != T):
            raise ValueError("Inconsistent extdata shape with epoched data")
        S = extdata.shape[1] if extdata.size != 0 else 0
    else:
        raise ValueError("data must be 2D or 3D")

    if N != 306:
        print("msfun_ica_meg_estimate - WARNING: Data may not come from Neuromag Elekta MEG system")

    print("msfun_ica_meg_estimate - Preparing data for ICA...")

    if epoching:
        print("msfun_ica_meg_estimate - Baseline correcting and concatenating epochs...")
        data = data - np.mean(data, axis=2, keepdims=True)
        data = msfun_sig_concat_epoch(data, K, "epochnum")

    IC = msfun_meg_ica_estimate(data, cfg.get("ica", {}))
    IC = msfun_meg_ica_cumulantanalysis(IC, cfg.get("cumulant", {}))

    if cfg.get("corranalysis", False):
        print("msfun_ica_meg_estimate - Preparing external signals for correlation analysis...")
        if epoching:
            print("msfun_ica_meg_estimate -       baseline correcting and concatenating epochs...")
            extdata_new = np.zeros((S, K * T))
            for k in range(K):
                av = np.mean(extdata[k, :, :], axis=1)
                extdata_new[:, k * T:(k + 1) * T] = extdata[k, :, :] - av[:, np.newaxis]
            extdata = extdata_new
        IC = msfun_meg_ica_corranalysis(IC, extdata, cfg.get("corr", {}))

    if epoching:
        print("msfun_ica_meg_estimate - Restoring epochs in IC time courses...")
        IC["S"] = msfun_sig_concat_epoch(IC["S"], K, "epochlength")

    if cfg.get("spectralanalysis", False):
        IC = msfun_meg_ica_powerspectrum(IC, cfg.get("fft", {}))
        IC = msfun_meg_ica_spectralanalysis(IC, cfg.get("spectral", {}))

    keep, reject, _ = msfun_meg_ica_resultfig(IC)
    cfg.setdefault("viewer", {})["list"] = reject
    msfun_meg_ica_viewer(IC, cfg["viewer"], raw)

    return IC
