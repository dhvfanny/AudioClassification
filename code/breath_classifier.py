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

if __name__ == '__main__':
    
    # Load breathing samples
    breath_files = get_file_locs(os.getcwd() + '\\data\\breathing', 'wav')
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
    nonbreath_files = get_file_locs(os.getcwd() + '\\data\\non_breathing', 'wav') 
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
    
    
    
    # Build a feature extraction pipeline
    fs = sr_b           # sampling rate for the whole pipeline
    window_length = 40  # in miliseconds
    n_mels = 20         # number of mel bands
    # First stage is a mel-frequency spectrogram
    MelSpec = FeatureExtractor(librosa.feature.melspectrogram, 
                                            n_fft = np.round(window_length * fs * 1e-3),
                                            hop_length = np.round(window_length * fs * 1e-3),
                                            n_mels = n_mels
                                            )
    # Second stage is log-amplitude; power is relative to peak in the signal
    LogAmp = FeatureExtractor(librosa.logamplitude, 
                                        ref_power=np.max)
    # Third stage transposes the data so that frames become samples
    Transpose = FeatureExtractor(np.transpose)
    # Last stage stacks all samples together into one matrix for training
    Stack = FeatureExtractor(np.vstack, iterate=False)
    breath_pipe = Pipeline([('Mel spectrogram', MelSpec), 
                                         ('Log amplitude', LogAmp),
                                         ('Transpose', Transpose),
                                         ('Stack', Stack)])
    
    # Apply feature extraction pipeline
    breath_feats = breath_pipe.fit_transform([b])
    nonbreath_feats = breath_pipe.fit_transform([nb])
    breath_targets = np.ones(breath_feats.shape[0])
    nonbreath_targets = np.zeros(nonbreath_feats.shape[0])
    all_feats = np.vstack((breath_feats, nonbreath_feats))
    all_targets = np.hstack((breath_targets, nonbreath_targets))
    
    # Split data into training and test
    X_train, X_test, y_train, y_test = train_test_split(all_feats, all_targets, test_size=0.1)
    
    # Train SVM classifier
    classifier = SVC(kernel = 'rbf', gamma = 0.005)                                    
    classifier.fit(X_train, y_train) 
    pred = classifier.predict(X_test)
    print("Accuracy:", 100 * np.sum(pred == y_test)/float(pred.shape[0]))

    
    