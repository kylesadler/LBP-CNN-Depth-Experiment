
""" run with 'python GNEG_classify.py feature_type classifier'
    ex: python GNEG_classify.py hog NN
    
"""

# import the necessary packages
import os
import h5py
import sys
import random
from keras.models import Sequential
from keras.layers import Dense, Conv2D

def main(feature_type, classifier_type):
        
    assert(feature_type in ['lbp3', 'lbp5', 'lbp7', 'none'])
        
    model = get_model(classifier_type)
    
    output_dir = feature_type+"_"+classifier_type
    if(os.path.exists(output_dir)):
        os.mkdir(output_dir)
        
    img_h5py = h5py.File(feature_type+".h5", 'r')
    label_h5py = h5py.File("label.h5", 'r')
    
    keys = random.Random(1).shuffle(list(h5py_file.keys()))
    train_index = int(len(keys*.64))
    val_index = int(len(keys*.8))
    
    
    checkpoint = ModelCheckpoint(
        output_dir,
        verbose=0,
        save_best_only=True
        )
    
    
    model.fit_generator(
        generate_data(img_h5py, label_h5py, keys[:train_index]),
        epochs=150,
        verbose=2,
        callbacks=[checkpoint],
        validation_data=generate_data(img_h5py, label_h5py, keys[train_index:val_index])
        )
    
    
    img_h5py.close()
    label_h5py.close()
    
    
def generate_data(img_h5py, label_h5py, keys):
    while True:
        for key in keys:
            yield(img_h5py[key], label_h5py[key])
    
    
def get_model(model_type):
    if(model_type == "nn"):
        return get_CNN(0)
    elif(model_type == "cnn1"):
        return get_CNN(1)
    elif(model_type == "cnn3"):
        return get_CNN(3)
    elif(model_type == "cnn5"):
        return get_CNN(5)
    else:
        raise AssertionError
    
    
def get_CNN(cnn_layers):
    model = Sequential()
    
    if(cnn_layers > 0):
        model.add(conv2D(256, input_dim=(256,256), activation='relu'))
        for i in range(cnn_layers-1):
            model.add(conv2D(256, activation='relu'))
    
    model.add(Dense(256, activation='relu'))
    model.add(Dense(30), activation='relu')
    model.add(Dense(2), activation='softmax')

    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    return model
    

if(__name__ == "__main__"):
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"#sys.argv[1] #"0" # "0, 1" for multiple
    main(sys.argv[1], sys.argv[2])
