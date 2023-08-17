#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from datetime import datetime
from dateutil import tz
import pytz
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.signal as sg

def plot_signal_and_spectrogram(data, title, fs, fmin, fmax, npseg):
  x = data
  f, t, Sxx = sg.spectrogram(x, fs)

  plt.figure(figsize=(12,5))
  plt.title(title)

  plt.subplot(1, 2, 1) # row 1, col 2 index 1
  plt.plot(x)
  plt.title(title)
  plt.ylabel('Magnitude')
  plt.xlabel('Time [sec]')

  plt.subplot(1, 2, 2) # index 2
  plt.pcolormesh(t, f[int(fmin*npseg/fs):int(fmax*npseg/fs)+1], np.abs(Sxx[int(fmin*npseg/fs):int(fmax*npseg/fs)+1,:]), shading='gouraud')
  plt.title("Spectrogram")
  plt.ylabel('Frequency [Hz]')
  plt.xlabel('Time [sec]')

  plt.show()
