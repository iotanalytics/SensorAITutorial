#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%% function script

import numpy as np
import math
import matplotlib.pyplot as plt
from utils import *

## Mount google drive
def fft_denoise(signal, fs, threshold):
  n = len(signal) #1000
  dt = fs
  fhat = np.fft.fft(signal, n)
  psd = fhat * np.conjugate(fhat)/n
  freq = (1/(dt*n)) * np.arange(n)
  L = np.arange(1, np.floor(n/2), dtype = 'int')
  indices = psd > threshold
  pdsclean = psd * indices
  fhat= indices * fhat
  ffilt = np.fft.ifft(fhat)
  ffilt = ffilt.real
  return ffilt



#%% main script
if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from signal_generate import * 

    length_seconds = 10
    sampling_rate = 100
    frequencies = [1, 3]
    data = generate_sine_waves(length_seconds, sampling_rate, frequencies, func="sin", add_noise=0.5, plot=False)

    plt.figure(figsize=(12,5))

    fs = sampling_rate
    threshold = 5
    denoise = fft_denoise(data, fs, threshold)

    fs = 100
    fmin = 0
    fmax = 10
    npseg = 256 #256

    # def plot_signal_and_spectrogram(data, title, fs, fmin, fmax, npseg):
    title = "Raw Data"
    x = data
    plot_signal_and_spectrogram(x, title, fs, fmin, fmax, npseg)

    x = denoise
    title = "Denoised Data"
    plot_signal_and_spectrogram(x, title, fs, fmin, fmax, npseg)

# %%
