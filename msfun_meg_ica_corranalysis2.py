import numpy as np
from scipy.fft import fft, ifft
from scipy.stats import pearsonr
from msfun_prepare_cosine_filter import msfun_prepare_cosine_filter

def msfun_meg_ica_corranalysis2(IC, extdata, cfg):
    if extdata is None or len(extdata) == 0:
        print('msfun_meg_ica_corranalysis2 - WARNING : No external data supplied... Skipping the correlation analysis.')
        return IC

    if 'S' not in IC or not isinstance(IC['S'], np.ndarray) or IC['S'].ndim not in [2, 3]:
        raise ValueError('msfun_meg_ica_corranalysis2 - ERROR : IC structure missing elements or inconsistent.')

    if IC['S'].ndim == 2:
        epoching = False
        numofic, T = IC['S'].shape
    else:
        epoching = True
        K, numofic, T = IC['S'].shape

    extdata = np.asarray(extdata)
    if extdata.ndim != IC['S'].ndim:
        raise ValueError('msfun_meg_ica_corranalysis2 - ERROR : External data array inconsistent.')

    if epoching:
        if extdata.shape != (K, extdata.shape[1], T):
            raise ValueError('msfun_meg_ica_corranalysis2 - ERROR : External data not consistent with IC.')
        S = extdata.shape[1]
    else:
        if extdata.shape[1] != T:
            raise ValueError('msfun_meg_ica_corranalysis2 - ERROR : External data not consistent with IC.')
        S = extdata.shape[0]

    cfg = cfg or {}
    cfg.setdefault('extname', [f'ext{k+1}' for k in range(S)])
    cfg.setdefault('filter', True)
    cfg.setdefault('Tsigcorr', 0.1)
    cfg.setdefault('Tpowcorr', 0.2)

    if cfg['filter']:
        filt = cfg.get('filt', {})
        filt.setdefault('sfreq', 1000)
        filt.setdefault('win', 'boxcar')
        filt.setdefault('par', ['high', 'low'])
        filt.setdefault('freq', [1, 25])
        filt.setdefault('width', [0.5, 5])
        cfg['filt'] = filt

    if epoching:
        print('msfun_meg_ica_corranalysis2 - Baseline correcting and concatenating epochs...')
        X = np.zeros((numofic, K * T))
        Y = np.zeros((S, K * T))
        for k in range(K):
            avX = np.mean(IC['S'][k, :, :], axis=1, keepdims=True)
            avY = np.mean(extdata[k, :, :], axis=1, keepdims=True)
            X[:, k*T:(k+1)*T] = IC['S'][k, :, :] - avX
            Y[:, k*T:(k+1)*T] = extdata[k, :, :] - avY
        IC['S'] = X
        extdata = Y
        T *= K

    if cfg['filter']:
        print('msfun_meg_ica_corranalysis2 - Filtering ICs and external data...')
        win, F = msfun_prepare_cosine_filter(cfg['filt'], T, cfg['filt']['sfreq'])
        IC['S'] = np.real(ifft(fft(IC['S'] * win, axis=1) * F, axis=1))
        extdata = np.real(ifft(fft(extdata * win, axis=1) * F, axis=1))

    print('msfun_meg_ica_corranalysis2 - Performing correlation analysis...')
    sigrho = np.corrcoef(extdata, IC['S'])[:S, S:]
    powrho = np.corrcoef(extdata**2, IC['S']**2)[:S, S:]

    if 'corr' not in IC:
        IC['corr'] = {}
    IC['corr']['list'] = []

    for k in range(S):
        extname = cfg['extname'][k]
        IC['corr'][f'sig_IC_{extname}'] = sigrho[k]
        IC['corr'][f'pow_IC_{extname}'] = powrho[k]
        IC['corr']['list'].extend(np.where(np.abs(sigrho[k]) >= cfg['Tsigcorr'])[0])
        IC['corr']['list'].extend(np.where(np.abs(powrho[k]) >= cfg['Tpowcorr'])[0])

    IC['corr']['list'] = sorted(set(IC['corr']['list']))

    print('msfun_meg_ica_corranalysis2 - Done.')
    return IC