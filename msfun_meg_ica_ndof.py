import numpy as np
import matplotlib.pyplot as plt

def msfun_meg_ica_ndof(data, cfg):
    if not isinstance(data, np.ndarray) or data.ndim not in [2, 3]:
        raise ValueError("msfun_meg_ica_ndof - ERROR: data must be a numeric array... Try again.")

    if data.ndim == 3:
        epoching = True
        K, N, T = data.shape
    else:
        epoching = False
        N, T = data.shape

    if not isinstance(cfg, dict):
        raise ValueError("msfun_meg_ica_ndof - ERROR: Configuration not a structure... Try again.")

    normalize = cfg.get("normalize", None)
    if normalize is not None:
        normalize = np.asarray(normalize)
        if normalize.ndim != 1 or len(normalize) != N or np.any(normalize <= 0):
            raise ValueError("msfun_meg_ica_ndof - ERROR: Normalization factors inconsistent... Try again.")
    else:
        normalize = None

    method = cfg.get("method", "rel")
    if method not in ["abs", "maxrel", "rel"]:
        raise ValueError("msfun_meg_ica_ndof - ERROR: Method not recognized... Try again.")

    param = cfg.get("param", 1e3)

    if epoching:
        print("msfun_meg_ica_ndof - Baseline correcting and concatenating epochs...")
        X = np.zeros((N, K*T))
        for k in range(K):
            av = np.mean(data[k, :, :], axis=1, keepdims=True)
            X[:, k*T:(k+1)*T] = data[k, :, :] - av
        data = X

    print("msfun_meg_ica_ndof - Normalizing data...")
    if normalize is None:
        normalize = np.std(data, axis=1)
        print("msfun_meg_ica_ndof -   using data standard deviation...")
    data = data / normalize[:, np.newaxis]

    print("msfun_meg_ica_ndof - Computing normalized data covariance and its eigenvalues...")
    D = np.linalg.eigvalsh(np.cov(data))
    D = np.sort(D)

    if method == "abs":
        cutoff = param
        n = np.argmax(D >= cutoff)
    elif method == "maxrel":
        cutoff = np.max(D) / param
        n = np.argmax(D >= cutoff)
    else:  # method == "rel"
        R = D[1:] / D[:-1]
        n = np.where(R >= param)[0][-1]

    ndof = N - n
    print(f"msfun_meg_ica_ndof - Estimated {ndof} largest eigendirections...")

    # Plot eigenvalues
    plt.figure("msfun_meg_ica_ndof - Normalized data covariance eigenvalues")
    if method in ["abs", "maxrel"]:
        plt.plot(D, "r")
        plt.xlabel("n")
        plt.ylabel("eigenvalue(n)")
        plt.plot(range(n+1, N), D[n+1:], "b")
        plt.axhline(y=cutoff, color="g")
    else:
        plt.subplot(1, 2, 1)
        plt.plot(D, "r")
        plt.xlabel("n")
        plt.ylabel("eigenvalue(n)")
        plt.plot(range(n+1, N), D[n+1:], "b")
        plt.subplot(1, 2, 2)
        plt.plot(R, "r")
        plt.xlabel("n")
        plt.ylabel("eigenvalue(n+1)/eigenvalue(n)")
        plt.plot(range(n+1, N-1), R[n+1:], "b")
        plt.axhline(y=param, color="g")
    plt.show()

    go = input(f"Keep {ndof} largest eigendirections? [y/n] ")
    if go.lower() in ["n", "no"]:
        ndof = int(input("Enter number of eigendirections to keep: "))

    return ndof