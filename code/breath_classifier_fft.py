'''
Author         : Oguzhan Gencoglu
Contact        : oguzhan.gencoglu@tut.fi
Created        : 11.07.2015
Latest Version : 21.07.2015

Train a classifier for breath detection
'''

from __future__ import absolute_import
from __future__ import print_function
from librosa.util import FeatureExtractor
from get_file_locs import get_file_locs
import librosa
import os
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sliding_window import sliding_window

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

if __name__ == '__main__':
    
    # Load breathing samples
    breath_files = get_file_locs('..\\data\\breathing', 'wav')
    b = np.array([])
    b_dur = 0
    sr_b = 22050
    for i in range(len(breath_files)):
        print('\nReading file (breathing) number', str(i + 1))
        breath_file = breath_files[i]
        temp_b, temp_sr_b = librosa.load(breath_file)
        temp_dur = temp_b.shape[0] / float(temp_sr_b)
        b_dur = b_dur + temp_dur
        print('\tFile (breathing) sampling rate :', str(temp_sr_b))
        print('\tFile (breathing) duration :', "{0:.2f}".format(temp_dur), 'seconds')
        b = np.append(b, temp_b)
    print('\n\tTotal duration (breathing) :', "{0:.2f}".format(b_dur), 'seconds')
    
    # Load non-breathing samples 
    nonbreath_files = get_file_locs('..\\data\\non_breathing', 'wav') 
    nb = np.array([])
    nb_dur = 0
    for j in range(len(nonbreath_files)):
        print('\nReading file (non-breathing) number', str(j + 1))
        nonbreath_file = nonbreath_files[j]
        temp_nb, temp_sr_nb = librosa.load(nonbreath_file, duration = 8) # too much nonbreath
        temp_dur = temp_nb.shape[0] / float(temp_sr_nb)
        nb_dur = nb_dur + temp_dur
        print('\tFile (non-breathing) sampling rate :', str(temp_sr_nb))
        print('\tFile (non-breathing) duration :', "{0:.2f}".format(temp_dur), 'seconds')
        nb = np.append(nb, temp_nb)  
    print('\n\tTotal duration (non-breathing) :', "{0:.2f}".format(nb_dur), 'seconds') 
    
    # windowing
    window_len = 1024
    b_feat = sliding_window(b, window_len)
    nb_feat = sliding_window(nb, window_len)
    
    # zero mean scaling within each window
    b_feat = b_feat - np.transpose(np.tile(np.mean(b_feat, axis = 1), (b_feat.shape[1], 1)))
    nb_feat = nb_feat - np.transpose(np.tile(np.mean(nb_feat, axis = 1), (nb_feat.shape[1], 1)))
    
    # fft features
    b_feat = np.abs(np.fft.fft(sliding_window(b, window_len), axis = 1))
    nb_feat = np.abs(np.fft.fft(sliding_window(nb, window_len), axis = 1))
    all_feats = np.vstack((b_feat, nb_feat))
    
    # create targets
    breath_targets = np.ones(b_feat.shape[0])
    nonbreath_targets = np.zeros(nb_feat.shape[0])
    all_targets = np.hstack((breath_targets, nonbreath_targets))
    
    # Split data into training and test
    X_train, X_test, y_train, y_test = train_test_split(all_feats, all_targets, test_size=0.1)
    
    # Train SVM classifier
    classifier = SVC(kernel = 'rbf', gamma = 0.005)                                    
    classifier.fit(X_train, y_train) 
    pred = classifier.predict(X_test)
    print("Accuracy:", 100 * np.sum(pred == y_test)/float(pred.shape[0]))    
    