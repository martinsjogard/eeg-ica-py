import numpy as np
from sklearn.decomposition import FastICA

def msfun_estimate_ica_meeg(sig, cfg=None):
    """
    Decomposes the [N x T] multivariate signal matrix using temporal ICA.

    Parameters:
    - sig: numpy.ndarray of shape [N x T]
    - cfg: dict with optional keys:
        - 'normalize': numpy.ndarray of shape [N] for normalization factors
        - 'fastica': dict of parameters to pass to FastICA (e.g., {'fun': 'tanh', 'n_components': 30})

    Returns:
    - IC: dict with keys:
        - 'A': Mixing matrix [N x numOfIC]
        - 'W': Unmixing matrix [numOfIC x N]
        - 'S': Independent components [numOfIC x T]
    """
    if sig.ndim != 2:
        raise ValueError("sig must be a 2D array [N x T]")

    N, T = sig.shape
    cfg = cfg or {}
    normalize = cfg.get("normalize", None)
    fastica_params = cfg.get("fastica", {'fun': 'tanh', 'n_components': min(N, 30)})

    # Signal normalization
    print("msfun_estimate_ica_meeg - Normalizing data units...")
    if normalize is None:
        print("msfun_estimate_ica_meeg -       using temporal standard deviation of data...")
        normalize = np.std(sig, axis=1)
    else:
        normalize = np.asarray(normalize)
        if normalize.ndim != 1 or normalize.shape[0] != N or np.any(normalize <= 0):
            raise ValueError("normalize must be a positive vector of length N")

    sig_norm = sig / normalize[:, np.newaxis]

    # Run FastICA
    print("msfun_estimate_ica_meeg - FASTICA in action...")
    ica = FastICA(**fastica_params)
    S = ica.fit_transform(sig_norm.T).T  # components x time
    A = ica.mixing_                      # N x components
    W = ica.components_                  # components x N

    # Restore original units
    print("msfun_estimate_ica_meeg - Restoring original data units...")
    A_scaled = A * normalize[:, np.newaxis]
    W_scaled = W / normalize[np.newaxis, :]

    print("msfun_eeg_ica_estimate - Done.")

    return {'A': A_scaled, 'W': W_scaled, 'S': S}
