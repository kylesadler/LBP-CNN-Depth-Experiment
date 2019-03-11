#!/bin/bash

# crop dataset
python GNEG_crop.py

# classify none
python GNEG_train.py none nn
python GNEG_train.py none cnn1
python GNEG_train.py none cnn3
python GNEG_train.py none cnn5

# classify lbp3
python GNEG_extract_features.py lbp3

python GNEG_train.py lbp3 nn
python GNEG_train.py lbp3 cnn1
python GNEG_train.py lbp3 cnn3
python GNEG_train.py lbp3 cnn5

# classify lbp5
python GNEG_extract_features.py lbp5

python GNEG_train.py lbp5 nn
python GNEG_train.py lbp5 cnn1
python GNEG_train.py lbp5 cnn3
python GNEG_train.py lbp5 cnn5

# classify lbp7
python GNEG_extract_features.py lbp7

python GNEG_train.py lbp7 nn
python GNEG_train.py lbp7 cnn1
python GNEG_train.py lbp7 cnn3
python GNEG_train.py lbp7 cnn5
