#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%% function script
import numpy as np
import math
import matplotlib.pyplot as plt

#%% main script
if __name__ == '__main__':

    from signal_generate import * 
    import numpy as np
    import matplotlib.pyplot as plt
    import ewtpy

    data  = generate_sine_waves(length_seconds=10, sampling_rate=100, frequencies=[1,3,6], add_noise=0, plot=False)

    ewt, mfb, boundaries = ewtpy.EWT1D(data, N = 3)

    plt.figure(figsize=(16,2))
    plt.title('Original signal')
    plt.xlabel('time (s)')
    plt.plot(data)
    plt.show()

    for i in range(ewt.shape[1]):
        plt.figure(figsize=(16,2))
        plt.xlabel("Time [s]")
        plt.ylabel("Mode %i" %(i))
        plt.locator_params(axis='y', nbins=5)
        plt.plot(ewt[:,i], 'g')
        plt.show()
# %%
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import signal

    from sklearn.decomposition import FastICA, PCA

    # #############################################################################
    # Generate sample data
    # np.random.seed(0)
    # n_samples = 2000
    # time = np.linspace(0, 8, n_samples)

    # s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
    # s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
    # s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

    s1  = generate_sine_waves(length_seconds=10, sampling_rate=100, frequencies=[1], add_noise=0.0, plot=False)
    s2  = generate_sine_waves(length_seconds=10, sampling_rate=100, frequencies=[3], add_noise=0.0, plot=False)
    s3  = generate_sine_waves(length_seconds=10, sampling_rate=100, frequencies=[5], add_noise=0.0, plot=False)

    S = np.c_[s1, s2, s3]
    S += 0.2 * np.random.normal(size=S.shape)  # Add noise

    S /= S.std(axis=0)  # Standardize data
    # Mix data
    A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
    X = np.dot(S, A.T)  # Generate observations

    # X = data
    # Compute ICA
    ica = FastICA(n_components=3)
    S_ = ica.fit_transform(X)  # Reconstruct signals
    A_ = ica.mixing_  # Get estimated mixing matrix

    # We can `prove` that the ICA model applies by reverting the unmixing.
    assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)

    # For comparison, compute PCA
    pca = PCA(n_components=3)
    H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components

    # #############################################################################
    # Plot results

    # plt.figure()

    models = [X, S, S_, H]
    names = ['Observations (mixed signal)',
            'True Sources',
            'ICA recovered signals',
            'PCA recovered signals']
    colors = ['red', 'steelblue', 'orange']

    num_rows = len(colors)
    num_cols = 1
    for ii, (model, name) in enumerate(zip(models, names), 1):
        # plt.subplot(4, 1, ii)
        plt.figure(figsize=(16,8))
        plt.title(name)
        ind = 1
        for sig, color in zip(model.T, colors):
            plt.subplot(num_rows, num_cols, ind) # row, col, index
            ind+=1
            plt.plot(sig, color=color)
        plt.tight_layout()
        plt.show()
    

# %%