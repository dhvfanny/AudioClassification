'''
Author         : Oguzhan Gencoglu
Contact        : oguzhan.gencoglu@tut.fi
Created        : 08.10.2014
Latest Version : 08.10.2014
'''

import numpy as np

def shuffle_data(data, targets):
    # Shuffles the data and targets
    
    shuffled_index = np.arange(data.shape[0]) 
    np.random.shuffle(shuffled_index)
    data = data[shuffled_index, :]
    if len(targets.shape) > 1:
        targets = targets[shuffled_index, :]
    else:
        targets = targets[shuffled_index]
        
    return data, targets