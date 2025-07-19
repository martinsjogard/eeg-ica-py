import numpy as np
import matplotlib.pyplot as plt

def msfun_meg_ica_resultfig(IC):
    reject = []

    # Handle cumulant analysis rejections
    Ncumulant = 0
    if 'cumulant' in IC:
        Ncumulant += int('skew' in IC['cumulant']) + int('kurt' in IC['cumulant'])
        reject.extend(IC['cumulant'].get('list', []))

    # Handle correlation analysis rejections
    Ncorr = 0
    corrnames = []
    if 'corr' in IC:
        corrnames = [k for k in IC['corr'].keys() if k not in ['list', 'Tcorr']]
        Ncorr = len(corrnames)
        reject.extend(IC['corr'].get('list', []))

    # Handle spectral analysis rejections
    Nspectral = 0
    if 'spectral' in IC and 'gof' in IC['spectral']:
        Nspectral = IC['spectral']['gof'].shape[1]
        reject.extend(IC['spectral'].get('list', []))

    reject = sorted(set(reject))
    numofic = IC['A'].shape[1]
    keep = [i for i in range(numofic) if i not in reject]

    # Determine subplot layout
    Ntot = Ncumulant + Ncorr + Nspectral
    ncol = int(np.floor(np.sqrt(Ntot)))
    nrow = int(np.ceil(Ntot / ncol))

    print("msfun_meg_ica_resultfig - Generating plots...")
    fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
    axs = axs.flatten()
    n = 0

    # Cumulant analysis plots
    if 'cumulant' in IC:
        if 'skew' in IC['cumulant']:
            ax = axs[n]; n += 1
            ax.plot(IC['cumulant']['skew'], 'x')
            ax.plot(reject, [IC['cumulant']['skew'][i] for i in reject], 'or')
            ax.axhline(IC['cumulant']['Tskew'], linestyle='--', color='g')
            ax.axhline(-IC['cumulant']['Tskew'], linestyle='--', color='g')
            ax.set_title("CUMULANT ANALYSIS : SKEWNESS")
            ax.set_xlabel("IC index"); ax.set_ylabel("skew"); ax.grid(True)
        if 'kurt' in IC['cumulant']:
            ax = axs[n]; n += 1
            ax.plot(IC['cumulant']['kurt'], 'x')
            ax.plot(reject, [IC['cumulant']['kurt'][i] for i in reject], 'or')
            ax.axhline(IC['cumulant']['Tkurt'], linestyle='--', color='g')
            ax.set_title("CUMULANT ANALYSIS : KURTOSIS")
            ax.set_xlabel("IC index"); ax.set_ylabel("kurt"); ax.grid(True)

    # Correlation analysis plots
    for k, name in enumerate(corrnames):
        ax = axs[n]; n += 1
        ax.plot(IC['corr'][name], 'x')
        ax.plot(reject, [IC['corr'][name][i] for i in reject], 'or')
        ax.axhline(IC['corr']['Tcorr'], linestyle='--', color='g')
        ax.axhline(-IC['corr']['Tcorr'], linestyle='--', color='g')
        ax.set_title(f"CORR ANALYSIS : IC/{name}")
        ax.set_xlabel("IC index"); ax.set_ylabel("corr"); ax.grid(True)

    # Spectral analysis plots
    for k in range(Nspectral):
        ax = axs[n]; n += 1
        ax.plot(IC['spectral']['gof'][:, k], 'x')
        ax.plot(reject, [IC['spectral']['gof'][i, k] for i in reject], 'or')
        ax.axhline(IC['spectral']['Tgof'][k], linestyle='--', color='g')
        label = IC['spectral']['fit'][2*k - 1]
        rng = IC['spectral']['fit'][2*k]
        ax.set_title(f"SPECTRAL ANALYSIS : {label} on [{rng[0]},{rng[1]}]")
        ax.set_xlabel("IC index"); ax.set_ylabel("gof"); ax.grid(True)

    plt.tight_layout()
    return keep, reject, fig