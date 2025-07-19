import numpy as np

def msfun_ica_meg_spectraldensity(IC, cfg):
    if not isinstance(IC, dict) or 'S' not in IC or not isinstance(IC['S'], np.ndarray):
        raise ValueError("msfun_ica_meg_spectraldensity - ERROR: IC structure missing elements or inconsistent... Try again.")

    S = IC['S']
    if S.ndim == 2:
        epoching = False
        numofic, T = S.shape
    elif S.ndim == 3:
        epoching = True
        K, numofic, T = S.shape
    else:
        raise ValueError("msfun_ica_meg_spectraldensity - ERROR: IC.S must be 2D or 3D array.")

    sfreq = cfg.get("sfreq", 1000)
    epoch_len = cfg.get("epoch", T)
    overlap = cfg.get("overlap", 2)

    if not isinstance(sfreq, (int, float)) or sfreq <= 0:
        raise ValueError("msfun_ica_meg_spectraldensity - ERROR: Sampling frequency must be a positive number.")
    if not isinstance(epoch_len, int) or epoch_len <= 0:
        raise ValueError("msfun_ica_meg_spectraldensity - ERROR: Epoch length must be a positive integer.")
    if not isinstance(overlap, int) or overlap <= 0:
        raise ValueError("msfun_ica_meg_spectraldensity - ERROR: Overlap number must be a positive integer.")

    print("msfun_ica_meg_spectraldensity - Computing Fourier power spectrum of ICs...")
    print(f"msfun_ica_meg_spectraldensity -       epoch length {epoch_len/sfreq} sec...")
    print(f"msfun_ica_meg_spectraldensity -       epochs overlap {overlap}...")

    if not epoching:
        powspctrm = np.zeros((numofic, epoch_len))
        ti = 0
        tf = ti + epoch_len
        nav = 0
        while tf <= T:
            epoch = S[:, ti:tf]
            powspctrm += np.abs(np.fft.fft(epoch, axis=1))**2
            nav += 1
            ti += round(epoch_len / overlap)
            tf = ti + epoch_len
        powspctrm /= nav
    else:
        powspctrm = np.zeros((K, numofic, epoch_len))
        ti = 0
        tf = ti + epoch_len
        nav = 0
        while tf <= T:
            epoch = S[:, :, ti:tf]
            powspctrm += np.abs(np.fft.fft(epoch, axis=2))**2
            nav += 1
            ti += round(epoch_len / overlap)
            tf = ti + epoch_len
        powspctrm = np.mean(powspctrm / nav, axis=0)

    freq = sfreq * np.arange(epoch_len // 2) / epoch_len
    IC['powspctrm'] = powspctrm[:, :epoch_len // 2] / epoch_len
    IC['freq'] = freq

    print(f"msfun_ica_meg_spectraldensity -       frequency domain [{freq[0]} {freq[-1]}] Hz...")
    print("msfun_ica_meg_spectraldensity - Done.")
    return IC
