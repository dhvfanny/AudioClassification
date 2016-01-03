'''
Author         : Oguzhan Gencoglu
Contact        : oguzhan.gencoglu@tut.fi
Created        : 15.10.2014
Latest Version : 19.10.2014
'''

import numpy as np
import copy

def cv(number_of_folds, fold_number, data, targets, val_test_param = 'both'):
    # N-fold cross validation, fold_number = 0,1,2,...
    
    # Get the number observations in each set
    number_of_folds = int(number_of_folds)
    number_of_data = data.shape[0]
    rem = int(number_of_data % number_of_folds)
    number_of_val = 0
    number_of_test = 0
    if val_test_param == 'both':
        number_of_val = int(np.floor(number_of_data / float(number_of_folds)))
        number_of_test = int(np.floor(number_of_data / float(number_of_folds)))
    elif val_test_param == 'val':
        number_of_val = int(np.floor(number_of_data / float(number_of_folds)))
    elif val_test_param == 'test':
        number_of_test = int(np.floor(number_of_data / float(number_of_folds)))
    else:
        print "ERROR: Wrong input for val_test_param!"
    if fold_number == number_of_folds - 1:
        if val_test_param in ['both', 'test']:
            number_of_test = number_of_test + rem
        else:
            number_of_val = number_of_val + rem

    # Initialize the masks
    false_array = np.zeros(number_of_data).astype(bool)
    train_bool = copy.deepcopy(false_array)
    val_bool = copy.deepcopy(false_array)
    test_bool = copy.deepcopy(false_array)
    
    test_bool[0:number_of_test] = True
    val_bool[number_of_test:number_of_test + number_of_val] = True
    train_bool[number_of_test + number_of_val:number_of_data] = True

    # Circular shift the sets depending on the number of folds
    shift_amount = np.max([number_of_val, number_of_test])
    train_bool = np.roll(train_bool, fold_number*shift_amount, axis=0)
    val_bool = np.roll(val_bool, fold_number*shift_amount, axis=0)
    test_bool = np.roll(test_bool, fold_number*shift_amount, axis=0)
    
    training_cv = data[train_bool,:]
    val_cv = data[val_bool,:]
    test_cv = data[test_bool,:]
    training_targets_cv = targets[train_bool]
    val_targets_cv = targets[val_bool]
    test_targets_cv = targets[test_bool]
    
    return training_cv, val_cv, test_cv, \
    training_targets_cv, val_targets_cv, test_targets_cv

''' Example:
    
if __name__ == '__main__':

    # Cross validate the data
    number_of_folds = 10
    for i in range(number_of_folds):    
        training_cv, val_cv, test_cv, training_targets_cv, val_targets_cv, \
        test_targets_cv = cv(number_of_folds, i, training_set, \
        training_targets, 'both')'''

