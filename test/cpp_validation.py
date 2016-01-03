# -*- coding: utf-8 -*-
'''
Author         : Oguzhan Gencoglu
Contact        : oguzhan.gencoglu@tut.fi
Created        : 03.01.2016
Latest Version : 03.01.2016

Compare C++ and python implementations.
'''

from __future__ import absolute_import
from __future__ import print_function
import librosa
import numpy as np


if __name__ == '__main__':
    
    # read data
    sig, temp_sr_b = librosa.load('unforced_front_30cm.wav')
    
    # first 1024 points (first window)
    sig = sig[0:1024]
    np.savetxt('unforced_front_30cm_first_window.txt', sig, delimiter='\n')
    
    # zero mean scaling
    sig = sig - np.mean(sig)
    np.savetxt('unforced_front_30cm_first_window_zero_mean.txt', sig, delimiter='\n')
    
    # feature extraction
    sig = np.abs(np.fft.fft(sig))
    np.savetxt('unforced_front_30cm_first_window_zero_mean_fft_magnitude.txt', sig, delimiter='\n')
    
    # save the result
    weights = np.loadtxt('weights.txt')
    intercept = np.loadtxt('intercept.txt')
    result = np.sum(sig*weights) + intercept
    np.savetxt('result.txt', np.array([result]))
    