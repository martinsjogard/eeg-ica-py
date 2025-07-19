import numpy as np
from sklearn.decomposition import FastICA

def msfun_ica_meg_decomp(sig, cfg=None):
    if cfg is None:
        cfg = {}

    if not isinstance(sig, np.ndarray) or sig.ndim != 2:
        raise ValueError("msfun_ica_meg_decomp - ERROR: Signal matrix inconsistent... Try again.")

    N, T = sig.shape

    if not isinstance(cfg, dict):
        raise ValueError("msfun_ica_meg_decomp - ERROR: Configuration not a structure... Try again.")

    normalize = cfg.get("normalize", None)
    if normalize is not None:
        normalize = np.asarray(normalize)
        if normalize.ndim != 1 or np.any(normalize <= 0) or len(normalize) != N:
            raise ValueError("msfun_ica_meg_decomp - ERROR: Normalization factors inconsistent... Try again.")
        if normalize.shape[0] != N:
            normalize = normalize.T
    else:
        print("msfun_ica_meg_decomp - using temporal standard deviation of data...")
        normalize = np.std(sig, axis=1)

    fastica_params = cfg.get("fastica", {"fun": "tanh", "max_iter": 200, "random_state": 0})
    if not isinstance(fastica_params, dict):
        raise ValueError("msfun_ica_meg_decomp - ERROR: FastICA parameters must be given in a dictionary... Try again.")

    print("msfun_ica_meg_decomp - Normalizing data units...")
    sig_norm = sig / normalize[:, np.newaxis]

    print("msfun_ica_meg_decomp - FASTICA in action...")
    ica = FastICA(**fastica_params)
    S = ica.fit_transform(sig_norm.T).T
    A = ica.mixing_
    W = ica.components_

    print("msfun_ica_meg_decomp - Restoring original data units...")
    A = A * normalize[:, np.newaxis]
    W = W / normalize[np.newaxis, :]

    IC = {
        "S": S,
        "A": A,
        "W": W
    }

    print("msfun_ica_meg_decomp - Done.")
    return IC
