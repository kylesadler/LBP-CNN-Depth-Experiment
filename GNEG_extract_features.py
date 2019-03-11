""" run as python GNEG_extract_features.py feature
    ex: python GNEG_extract_features.py lbp3
"""

# import the necessary packages
from skimage import feature
import os
import cv2
import h5py
from math import *
import numpy as np
import sys

def main(feature_type):
    h5py_file_in = h5py.File("none.h5", 'r')
    
    for key in list(h5py_file_in.keys()):
        
        current_batch = []

        for img in h5py_file_in[key]: # for each img in batch
            assert(len(img.shape) == 2)
            
            if(feature_type == "lbp3"):
                feature_map = feature.local_binary_pattern(img, 8, 3, method='uniform')
            elif(feature_type == "lbp5"):
                feature_map = feature.local_binary_pattern(img, 8, 5, method='uniform')
            elif(feature_type == "lbp7"):
                feature_map = feature.local_binary_pattern(img, 8, 7, method='uniform')
            else:
                raise
            
            current_batch.append(feature_map)
            
        h5py_file_out = h5py.File(feature_type+".h5", 'a')
        h5py_file_out.create_dataset(key, data=np.asarray(current_batch))
        h5py_file_out.close()

    h5py_file_in.close()
    

if(__name__ == "__main__"):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"#sys.argv[1] #"0" # "0, 1" for multiple
    main(sys.argv[1])