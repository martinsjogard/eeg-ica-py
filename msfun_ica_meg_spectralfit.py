import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import Dict

def msfun_ica_meg_spectralfit(IC: Dict, cfg: Dict) -> Dict:
    if 'freq' not in IC or 'powspctrm' not in IC:
        raise ValueError("IC structure missing required fields 'freq' and 'powspctrm'")

    numofic, F = IC['powspctrm'].shape
    freq = IC['freq']
    IC['spectral'] = {}
    IC['spectral']['fit'] = cfg['fit']
    L = len(cfg['fit']) // 2
    gof = np.zeros((numofic, L))

    for k in range(L):
        kind = cfg['fit'][2*k].lower()
        fmin, fmax = cfg['fit'][2*k+1]
        f1 = np.where(freq >= fmin)[0][0]
        f2 = np.where(freq <= fmax)[0][-1]
        cfg['fit'][2*k+1] = [f1, f2]
        IC['spectral']['fit'][2*k+1] = freq[f1:f2+1].tolist()

    if 'Tgof' not in cfg:
        cfg['Tgof'] = [0.03] * L

    if cfg.get('visual', True):
        plt.ion()
        fig, axes = plt.subplots(2, L, figsize=(5*L, 8))

    for n in range(numofic):
        for k in range(L):
            fidx = cfg['fit'][2*k+1]
            x = IC['powspctrm'][n, fidx[0]:fidx[1]+1]
            f = freq[fidx[0]:fidx[1]+1]
            if cfg['fit'][2*k].lower() == 'linear':
                def model(f, a, b): return a + b * f
                popt, _ = curve_fit(model, f, x)
                xhat = model(f, *popt)
            elif cfg['fit'][2*k].lower() == 'powlaw':
                def model(f, a, b): return a * f ** b
                logf = np.log(f)
                logx = np.log(x)
                coef = np.polyfit(logf, logx, 1)
                a_init = np.exp(coef[1])
                b_init = coef[0]
                popt, _ = curve_fit(model, f, x, p0=[a_init, b_init])
                xhat = model(f, *popt)
            else:
                raise ValueError(f"Unsupported fit type: {cfg['fit'][2*k]}")
            gof[n, k] = np.sum((x - xhat) ** 2) / np.sum(x ** 2)

            if cfg.get('visual', True):
                ax1, ax2 = axes[0, k], axes[1, k]
                ax1.clear(); ax2.clear()
                ax1.plot(f, x, label="data")
                ax1.plot(f, xhat, 'r', label="fit")
                ax1.set_title(f"IC {n} - {cfg['fit'][2*k]} fit")
                ax1.legend()
                ax2.plot(f, (x - xhat) ** 2 / np.mean(x ** 2), 'rx')
                ax2.axhline(cfg['Tgof'][k], linestyle='--')
                ax2.set_title(f"error (mean = {gof[n,k]:.3f})")
                plt.pause(0.1)

    IC['spectral']['gof'] = gof.tolist()
    IC['spectral']['Tgof'] = cfg['Tgof']
    IC['spectral']['list'] = [i for i in range(numofic) if all(gof[i, j] < cfg['Tgof'][j] for j in range(L))]

    if cfg.get('visual', True):
        plt.ioff()
        plt.show()
    return IC
