# -*- coding: utf-8 -*-
'''
Author         : Oguzhan Gencoglu
Contact        : oguzhan.gencoglu@tut.fi
Created        : 11.07.2015
Latest Version : 03.01.2016

Train a classifier for breath detection
'''

from __future__ import absolute_import
from __future__ import print_function
from get_file_locs import get_file_locs
import librosa
import numpy as np
from sklearn.svm import LinearSVC
from sliding_window import sliding_window
from datetime import datetime
from cross_validate import cv
from shuffle_data import shuffle_data
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
    
    # Load breathing samples
    breath_files = get_file_locs('data\\breathing', 'wav')
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
    
    # Load non-breathing samples 
    nonbreath_files = get_file_locs('data\\non_breathing', 'wav') 
    nb = np.array([])
    nb_dur = 0
    for j in range(len(nonbreath_files)):
        print('\nReading file (non-breathing) number', str(j + 1))
        nonbreath_file = nonbreath_files[j]
        temp_nb, temp_sr_nb = librosa.load(nonbreath_file, duration = 8) # if limit is wanted, duration = 8
        temp_dur = temp_nb.shape[0] / float(temp_sr_nb)
        nb_dur = nb_dur + temp_dur
        print('\tFile (non-breathing) sampling rate :', str(temp_sr_nb))
        print('\tFile (non-breathing) duration :', "{0:.2f}".format(temp_dur), 'seconds')
        nb = np.append(nb, temp_nb)  
        
    print('\n\tTotal duration (breathing) :', "{0:.2f}".format(b_dur), 'seconds')
    print('\n\tTotal duration (non-breathing) :', "{0:.2f}".format(nb_dur), 'seconds') 
    
    # windowing
    window_len = 1024
    b_feat = sliding_window(b, window_len, window_len/2)
    nb_feat = sliding_window(nb, window_len, window_len/2)
    
    # zero mean scaling within each window
    b_feat = b_feat - np.transpose(np.tile(np.mean(b_feat, axis = 1), (b_feat.shape[1], 1)))
    nb_feat = nb_feat - np.transpose(np.tile(np.mean(nb_feat, axis = 1), (nb_feat.shape[1], 1)))
    
    '''# unity variance scaling
    b_feat = b_feat / np.transpose(np.tile(np.std(b_feat, axis = 1), (b_feat.shape[1], 1)))
    nb_feat = nb_feat / np.transpose(np.tile(np.std(nb_feat, axis = 1), (nb_feat.shape[1], 1)))'''  
    
    # fft features
    b_feat = np.abs(np.fft.fft(b_feat, axis = 1))
    nb_feat = np.abs(np.fft.fft(nb_feat, axis = 1))
    all_feats = np.vstack((b_feat, nb_feat))
    
    # create targets
    breath_targets = np.ones(b_feat.shape[0])
    nonbreath_targets = np.zeros(nb_feat.shape[0])
    all_targets = np.hstack((breath_targets, nonbreath_targets))
    
    # Shuffle and cross validate the data
    all_feats, all_targets = shuffle_data(all_feats, all_targets)
    #C_list = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    C_list = [0.1]
    all_accuracies = []
    for C in C_list:
        number_of_folds = 10
        print("Regularization parameter: ", C)
        start = datetime.now()
        accuracies = []
        for i in range(number_of_folds): 
            training_cv, _, test_cv, training_targets_cv, _, \
            test_targets_cv = cv(number_of_folds, i, all_feats, \
            all_targets, 'test')
        
            # Train SVM classifier
            classifier = LinearSVC(C = C, loss='squared_hinge')  # 'squared_hinge'                                 
            classifier.fit(training_cv, training_targets_cv) 
            pred = classifier.predict(test_cv)
            accuracy = 100 * classifier.score(test_cv, test_targets_cv)
            accuracies.append(accuracy)
            print("Accuracy:", accuracy)  
        stop = datetime.now()
        print("\n\n\t\tScript Running Time: %s"%str(stop - start))
        all_accuracies.append(np.mean(accuracies))
    print(zip(C_list, all_accuracies)) # 0.1 92.60
    
    # ROC curve
    all_feats, all_targets = shuffle_data(all_feats, all_targets)
    accuracies = []
    dfs = []
    for i in range(number_of_folds): 
        training_cv, _, test_cv, training_targets_cv, _, \
        test_targets_cv = cv(number_of_folds, i, all_feats, \
        all_targets, 'test')
    
        # Train SVM classifier
        classifier = LinearSVC(C = 0.1, loss='squared_hinge')  # 'squared_hinge'                                 
        classifier.fit(training_cv, training_targets_cv) 
        pred = classifier.predict(test_cv)
        accuracy = 100 * classifier.score(test_cv, test_targets_cv)
        accuracies.append(accuracy)
        print("Accuracy:", accuracy)
        dfs.append(classifier.decision_function(test_cv))
        break
    print(np.mean(accuracies))
    
    print(roc_auc_score(all_targets, np.concatenate(dfs)))
    fpr, tpr, thresholds = roc_curve(all_targets, np.concatenate(dfs), pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    sns.set_style("whitegrid")
    sns.set_context("paper")
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()        
        
    

    
    '''
    # print the details of the trained SVM
    classifier = LinearSVC(C = 0.2, loss='squared_hinge')  # 'squared_hinge'                                 
    classifier.fit(all_feats, all_targets) 
    coefs = classifier.coef_
    print("Weigths of features (for linear case):", coefs)
    intercept = classifier.intercept_[0]
    print("Intercept term:", intercept)
    
    # Save the trained weights
    np.savetxt('weights_new_data4.txt', coefs, delimiter='\n')
    np.savetxt('intercept_new_data4.txt', np.array([intercept]))
    '''