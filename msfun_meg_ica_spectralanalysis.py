import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

def msfun_meg_ica_spectralanalysis(IC: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(IC, dict) or 'freq' not in IC or 'powspctrm' not in IC:
        raise ValueError("IC structure missing elements or inconsistent")

    freq = np.array(IC['freq']).flatten()
    powspctrm = np.array(IC['powspctrm'])

    if powspctrm.shape[1] != len(freq):
        raise ValueError("Frequency and power spectrum dimensions do not match")

    numofic, F = powspctrm.shape

    cfg = cfg.copy()
    cfg['visual'] = cfg.get('visual', True)
    if 'fit' not in cfg or not isinstance(cfg['fit'], list) or len(cfg['fit']) % 2 != 0:
        raise ValueError("cfg.fit must be a list of curve/band pairs")

    L = len(cfg['fit']) // 2
    cfg['Tgof'] = np.array(cfg.get('Tgof', [0.03] * L))

    IC['spectral'] = {}
    IC['spectral']['fit'] = list(cfg['fit'])
    for k in range(L):
        fband = cfg['fit'][2*k+1]
        idx_start = np.searchsorted(freq, fband[0], side='left')
        idx_end = np.searchsorted(freq, fband[1], side='right') - 1
        IC['spectral']['fit'][2*k+1] = [freq[idx_start], freq[idx_end]]

    IC['spectral']['gof'] = np.zeros((numofic, L))

    if cfg['visual']:
        plt.ion()
        fig, axs = plt.subplots(2, L, figsize=(6*L, 8))

    for n in range(numofic):
        for k in range(L):
            fit_type = cfg['fit'][2*k]
            idx_start = np.searchsorted(freq, cfg['fit'][2*k+1][0], side='left')
            idx_end = np.searchsorted(freq, cfg['fit'][2*k+1][1], side='right')
            x = powspctrm[n, idx_start:idx_end]
            f = freq[idx_start:idx_end]

            if fit_type == 'linear':
                p = np.polyfit(f, x, 1)
                xhat = np.polyval(p, f)
            elif fit_type == 'powlaw':
                p = np.polyfit(np.log(f), np.log(x), 1)
                p = [np.exp(p[1]), p[0]]
                xhat = p[0] * f ** p[1]
            else:
                raise ValueError("Unsupported fit type")

            IC['spectral']['gof'][n, k] = np.sum((x - xhat)**2) / np.sum(x**2)

            if cfg['visual']:
                axs[0, k].plot(f, x, label=f'IC {n}')
                axs[0, k].plot(f, xhat, 'r')
                axs[0, k].set_title(f"{fit_type} fit on IC {n}")
                axs[1, k].plot(f, (x - xhat)**2 / np.mean(x**2), 'rx')
                axs[1, k].axhline(cfg['Tgof'][k], linestyle='--', color='k')
                axs[1, k].set_title(f"model error (mean = {IC['spectral']['gof'][n, k]:.3f})")
        if cfg['visual']:
            plt.pause(0.5)

    mask = np.ones((numofic,), dtype=bool)
    for k in range(L):
        mask = mask & (IC['spectral']['gof'][:, k] < cfg['Tgof'][k])
    IC['spectral']['list'] = np.where(mask)[0].tolist()
    IC['spectral']['Tgof'] = cfg['Tgof'].tolist()

    return IC