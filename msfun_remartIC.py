import numpy as np
import matplotlib.pyplot as plt

def msfun_remartIC(data, IC, list=None, flag=True):
    if list is None or len(list) == 0:
        if isinstance(IC, dict) and "artdetect" in IC and isinstance(IC["artdetect"], dict) and "list" in IC["artdetect"]:
            list = IC["artdetect"]["list"]
        else:
            raise ValueError("remartIC - ERROR : A list of artifactual ICs must be given either in IC.artdetect.list or in a third input vector list...")

    # Check data dimensionality
    if data.ndim == 2:
        if data.shape[0] != IC["A"].shape[0] or data.shape[1] != IC["S"].shape[1] or IC["S"].shape[0] != IC["A"].shape[1]:
            raise ValueError("remartIC - ERROR : Input data and IC inconsistent for 2D case.")
    elif data.ndim == 3:
        if data.shape[1] != IC["A"].shape[0] or data.shape[0] != IC["S"].shape[0] or data.shape[2] != IC["S"].shape[2] or IC["S"].shape[1] != IC["A"].shape[1]:
            raise ValueError("remartIC - ERROR : Input data and IC inconsistent for 3D case.")
    else:
        raise ValueError("remartIC - ERROR : First input data must be array with 2 or 3 dimensions.")

    data2 = np.copy(data)
    print(f"remartIC - Removing {len(list)} ICs...")

    if data.ndim == 2:
        data2 = data - IC["A"][:, list] @ IC["S"][list, :]
    else:
        for k in range(data.shape[0]):
            data2[k, :, :] = data[k, :, :] - IC["A"][:, list] @ IC["S"][k, list, :]

    if flag:
        print("remartIC - Generating comparative plots...")
        nfig = len(list)
        plt.figure(figsize=(10, 3 * nfig))
        if data.ndim == 2:
            for idx, ic in enumerate(list):
                chan = np.argmax(np.abs(IC["A"][:, ic]))
                plt.subplot(nfig, 1, idx + 1)
                plt.plot(data[chan, :], label="Original")
                plt.plot(data2[chan, :], 'r', label="Cleaned")
                plt.title(f"Channel {chan + 1} with maximal mixing for IC {ic + 1}")
                plt.axis("tight")
                plt.axis("off")
        else:
            for idx, ic in enumerate(list):
                chan = np.argmax(np.abs(IC["A"][:, ic]))
                diff = data[:, chan, :] - IC["S"][:, ic, :]
                trial = np.argmax(np.sum(diff**2, axis=1))
                plt.subplot(nfig, 1, idx + 1)
                plt.plot(data[trial, chan, :], label="Original")
                plt.plot(data2[trial, chan, :], 'r', label="Cleaned")
                plt.title(f"Channel {chan + 1}, trial {trial + 1} with max mixing for IC {ic + 1}")
                plt.axis("tight")
                plt.axis("off")
        plt.tight_layout()
        plt.show()

    print("remartIC - ICs removed from data.")
    return data2