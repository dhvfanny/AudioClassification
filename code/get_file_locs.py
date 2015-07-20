'''
Author         : Oguzhan Gencoglu
Contact        : oguzhan.gencoglu@tut.fi
Created        : 18.07.2015
Latest Version : 20.07.2015
'''

import glob
import os

def get_file_locs(directory, extension):
    # Returns a list of file paths of a given extension under a directory
    # E.g. get_file_locs('data\images', "png")
    
    files = []
    for root, dirnames, filenames in os.walk(directory):
        files.extend(glob.glob(root + "/*." + extension))

    return files