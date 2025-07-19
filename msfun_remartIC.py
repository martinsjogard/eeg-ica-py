import numpy as np
import matplotlib.pyplot as plt

def msfun_remartIC(data, IC, lst=None, flag=True):
    if lst is None or len(lst) == 0:
        if isinstance(IC, dict) and 'artdetect' in IC and 'list' in IC['artdetect']:
            lst = IC['artdetect']['list']
        else:
            raise ValueError("List of artifactual ICs must be provided or found in IC['artdetect']['list'].")

    data2 = np.copy(data)

    if data.ndim == 2:
        if data.shape[0] != IC['A'].shape[0] or data.shape[1] != IC['S'].shape[1] or IC['S'].shape[0] != IC['A'].shape[1]:
            raise ValueError("Input data and IC structure inconsistent for continuous data.")
        data2 = data - IC['A'][:, lst] @ IC['S'][lst, :]
    elif data.ndim == 3:
        if data.shape[1] != IC['A'].shape[0] or data.shape[0] != IC['S'].shape[0] or data.shape[2] != IC['S'].shape[2] or IC['S'].shape[1] != IC['A'].shape[1]:
            raise ValueError("Input data and IC structure inconsistent for epoched data.")
        for k in range(data.shape[0]):
            data2[k, :, :] = data[k, :, :] - IC['A'][:, lst] @ IC['S'][k, lst, :]
    else:
        raise ValueError("Data must be a 2D or 3D array.")

    if flag:
        nfig = len(lst)
        plt.figure(figsize=(10, 2 * nfig))
        if data.ndim == 2:
            for i, ic in enumerate(lst):
                chan = np.argmax(np.abs(IC['A'][:, ic]))
                plt.subplot(nfig, 1, i+1)
                plt.plot(data[chan, :], label='Original')
                plt.plot(data2[chan, :], 'r', label='Cleaned')
                plt.title(f"Channel {chan} for IC {ic}")
                plt.axis('tight')
                plt.axis('off')
        else:
            for i, ic in enumerate(lst):
                chan = np.argmax(np.abs(IC['A'][:, ic]))
                diffs = np.sum((data[:, chan, :] - IC['S'][:, ic, :])**2, axis=1)
                trial = np.argmax(diffs)
                plt.subplot(nfig, 1, i+1)
                plt.plot(data[trial, chan, :], label='Original')
                plt.plot(data2[trial, chan, :], 'r', label='Cleaned')
                plt.title(f"Channel {chan}, Trial {trial} for IC {ic}")
                plt.axis('tight')
                plt.axis('off')
        plt.tight_layout()
        plt.show()

    return data2