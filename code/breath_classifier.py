'''
Author         : Oguzhan Gencoglu
Contact        : oguzhan.gencoglu@tut.fi
Created        : 11.07.2015
Latest Version : 20.07.2015

Train a classifier for breath detection
'''

from __future__ import absolute_import
from __future__ import print_function
from librosa.util import FeatureExtractor
import librosa
import glob
from set_wd import set_wd
import os
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split

def get_file_locs(directory, extension):
    # Returns a list of file names of a given extension under a directory
    # E.g. get_file_locs(images, "png")
    
    files = []
    for root, dirnames, filenames in os.walk(directory):
        files.extend(glob.glob(root + "/*." + extension))

    return files

if __name__ == '__main__':
    
    project_dir = "Universite\\Research\\BreathDetection"
    set_wd(project_dir)
    
    # Load breathing samples
    breath_files = get_file_locs(os.getcwd() + '\\data\\breathing', 'wav')
    b = np.array([])
    b_dur = 0
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
    for j in range(len(breath_files)):
        print('\nReading file (non-breathing) number', str(j + 1))
        nonbreath_file = nonbreath_files[j]
        temp_nb, temp_sr_nb = librosa.load(nonbreath_file)
        temp_dur = temp_nb.shape[0] / float(temp_sr_nb)
        nb_dur = nb_dur + temp_dur
        print('\tFile (non-breathing) sampling rate :', str(temp_sr_nb))
        print('\tFile (non-breathing) duration :', "{0:.2f}".format(temp_dur), 'seconds')
        nb = np.append(nb, temp_nb)  
    print('\n\tTotal duration (non-breathing) :', "{0:.2f}".format(nb_dur), 'seconds')   
    
    # Build a feature extraction pipeline
    # First stage is a mel-frequency spectrogram
    MelSpec = FeatureExtractor(librosa.feature.melspectrogram, 
                                            n_fft=1024,
                                            hop_length = 512,
                                            n_mels=20,
                                            # fmax=librosa.midi_to_hz(116), 
                                            # fmin=librosa.midi_to_hz(24)
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
    
    breath_feats = breath_pipe.fit_transform([b])
    nonbreath_feats = breath_pipe.fit_transform([nb])
    breath_targets = np.ones(breath_feats.shape[0])
    nonbreath_targets = np.zeros(nonbreath_feats.shape[0])
    all_feats = np.vstack((breath_feats, nonbreath_feats))
    all_targets = np.hstack((breath_targets, nonbreath_targets))
    
    classifier = SVC(gamma=0.001)                                    
    X_train, X_test, y_train, y_test = train_test_split(all_feats, all_targets, test_size=0.1)
    
    # Train a classifier
    classifier.fit(X_train, y_train) 
    pred = classifier.predict(X_test)
    print("Accuracy:", 100 * np.sum(pred == y_test)/float(pred.shape[0]))

    
    