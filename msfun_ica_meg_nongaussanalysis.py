import numpy as np
from scipy.stats import skew, kurtosis

def msfun_ica_meg_nongaussanalysis(IC, cfg):
    if 'S' not in IC or not isinstance(IC['S'], np.ndarray) or IC['S'].ndim not in [2, 3]:
        raise ValueError('msfun_ica_meg_nongaussanalysis - ERROR : IC structure missing elements or inconsistent.')

    epoching = IC['S'].ndim == 3
    if epoching:
        K, numofic, T = IC['S'].shape
    else:
        numofic, T = IC['S'].shape

    if not isinstance(cfg, dict):
        raise ValueError('msfun_ica_meg_nongaussanalysis - Configuration must be a structure... Try again.')

    Tskew = cfg.get('Tskew', np.nan)
    Tkurt = cfg.get('Tkurt', 15)

    if epoching:
        print('msfun_ica_meg_nongaussanalysis - Baseline correcting and concatenating epochs...')
        X = np.zeros((numofic, K * T))
        for k in range(K):
            av = np.mean(IC['S'][k, :, :], axis=1, keepdims=True)
            X[:, k*T:(k+1)*T] = IC['S'][k, :, :] - av
        IC['S'] = X

    print('msfun_ica_meg_nongaussanalysis - Computing IC skewness and kurtosis...')
    skew_vals = skew(IC['S'], axis=1, bias=False)
    kurt_vals = kurtosis(IC['S'], axis=1, bias=False)  # excess kurtosis by default is True

    print('msfun_ica_meg_nongaussanalysis - Classifying ICs using their kurtosis...')
    sorted_indices = np.argsort(-kurt_vals)
    IC['S'] = IC['S'][sorted_indices, :]
    if 'A' in IC:
        IC['A'] = IC['A'][:, sorted_indices]
    if 'W' in IC:
        IC['W'] = IC['W'][sorted_indices, :]

    IC['cumulant'] = {
        'skew': skew_vals[sorted_indices],
        'kurt': kurt_vals[sorted_indices],
        'Tskew': Tskew,
        'Tkurt': Tkurt
    }

    list_skew = np.where(np.abs(IC['cumulant']['skew']) > Tskew)[0] if not np.isnan(Tskew) else np.array([], dtype=int)
    list_kurt = np.where(IC['cumulant']['kurt'] > Tkurt)[0] if not np.isnan(Tkurt) else np.array([], dtype=int)
    IC['cumulant']['list'] = np.sort(np.unique(np.concatenate((list_skew, list_kurt))))

    print('msfun_ica_meg_nongaussanalysis - Done.')
    return IC
