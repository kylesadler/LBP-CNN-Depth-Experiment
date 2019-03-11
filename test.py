
""" run with 'python GNEG_classify.py feature_classifier'
    ex: python GNEG_classify.py hog NN
    
"""

# import the necessary packages
import os
import h5py
import sys
import random
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Conv2D

def main(model_dir):
        
    #Load the saved model
    model = model_from_json(open(os.path.join(model_dir, 'architecture.json')).read())
    model.load_weights(os.path.join(model_dir, 'best_weights.h5'))
    
    img_h5py = h5py.File(feature_type+".h5", 'r')
    label_h5py = h5py.File("label.h5", 'r')
    
    keys = random.Random(1).shuffle(list(h5py_file.keys()))
    index = int(len(keys*.8))
    
    
    checkpoint = ModelCheckpoint(
        model_dir,
        verbose=0,
        save_best_only=True
        )
    
    
    model.predict_generator(
        generate_data(img_h5py, label_h5py, keys[index:]),
        verbose=2,
        callbacks=[checkpoint],
        )
    
    
    img_h5py.close()
    label_h5py.close()
    
    
def generate_data(img_h5py, label_h5py, keys):
    while True:
        for key in keys:
            yield(img_h5py[key], label_h5py[key])
    

if(__name__ == "__main__"):
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"#sys.argv[1] #"0" # "0, 1" for multiple
    main(sys.argv[1])
