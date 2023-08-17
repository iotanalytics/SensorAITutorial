#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import math
import matplotlib.pyplot as plt

#%% function script
def generate_sine_waves(length_seconds, sampling_rate, frequencies, func="sin", add_noise=0, plot=False):
    r"""
    Generate a n-D array, `length_seconds` seconds signal at `sampling_rate` sampling rate.
    Cited from https://towardsdatascience.com/hands-on-signal-processing-with-python-9bda8aad39de

    Args:
        length_seconds : float
            Duration of signal in seconds (i.e. `10` for a 10-seconds signal, `3.5` for a 3.5-seconds signal)
        sampling_rate : int
            The sampling rate of the signal.
        frequencies : 1 or 2 dimension python list a floats
            An array of floats, where each float is the desired frequencies to generate (i.e. [5, 12, 15] to generate a signal containing a 5-Hz, 12-Hz and 15-Hz)
            2 dimension python list, i.e. [[5, 12, 15],[1]], to generate a signal with 2 channels, where the second channel containing 1-Hz signal
        func : string, optional, default: sin
            The periodic function to generate signal, either `sin` or `cos`
        add_noise : float, optional, default: 0
            Add random noise to the signal, where `0` has no noise
        plot : boolean, optional, default: False
            Plot the generated signal

    Returns:
        signal : n-d ndarray
            Generated signal, a numpy array of length `sampling_rate*length_seconds`
    Usage:
        >>> s = generate_sine_waves(length_seconds=4,
        >>>     sampling_rate=100,
        >>>     frequencies=[2],
        >>>     plot=True
        >>> )
        >>>
        >>> s = generate_sine_waves(length_seconds=4,
        >>>     sampling_rate=100,
        >>>     frequencies=[1,2],
        >>>     func="cos",
        >>>     add_noise=0.5,
        >>>     plot=True
        >>> )
        >>>
        >>> s = generate_sine_waves(length_seconds=3.5,
        >>>     sampling_rate=100,
        >>>     frequencies=[[1,2],[1],[2]],
        >>>     plot=True
        >>> )
    """

    frequencies = np.array(frequencies, dtype=object)
    assert len(frequencies.shape) == 1 or len(frequencies.shape) == 2, "frequencies must be 1d or 2d python list"

    expanded = False
    if isinstance(frequencies[0], int):
        frequencies = np.expand_dims(frequencies, axis=0)
        expanded = True

    sampling_rate = int(sampling_rate)
    npnts = int(sampling_rate*length_seconds)  # number of time samples
    time = np.arange(0, npnts)/sampling_rate
    signal = np.zeros((frequencies.shape[0],npnts))

    for channel in range(0,frequencies.shape[0]):
        for fi in frequencies[channel]:
            if func == "cos":
                signal[channel] = signal[channel] + np.cos(2*np.pi*fi*time)
            else:
                signal[channel] = signal[channel] + np.sin(2*np.pi*fi*time)

        # normalize
        max = np.repeat(signal[channel].max()[np.newaxis], npnts)
        min = np.repeat(signal[channel].min()[np.newaxis], npnts)
        signal[channel] = (2*(signal[channel]-min)/(max-min))-1

    if add_noise:
        noise = np.random.uniform(low=0, high=add_noise, size=(frequencies.shape[0],npnts))
        signal = signal + noise

    if plot:
        plt.plot(time, signal.T)
        plt.title('Signal with sampling rate of '+str(sampling_rate)+', lasting '+str(length_seconds)+'-seconds')
        plt.xlabel('Time (sec.)')
        plt.ylabel('Amplitude')
        plt.show()

    if expanded:
        signal = signal[0]

    return signal

#%% main script
if __name__ == '__main__':

    import math
    import matplotlib.pyplot as plt
    import numpy as np

    data = generate_sine_waves(length_seconds=10,
             sampling_rate=100,
             frequencies=[1,3],
             add_noise=0.5,
             plot=False
         )

    print(data.shape)
    plt.figure(figsize=(20, 4))
    plt.title("simulated sine waves")
    # plt.plot (data1)
    # plt.plot (data2)
    plt.plot (data)
    plt.show()
# %%
