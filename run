#!/bin/bash

# crop dataset
python crop.py

# classify none
python train.py none nn
python test.py none_nn

python train.py none cnn1
python test.py none_cnn1

python train.py none cnn3
python test.py none_cnn3

python train.py none cnn5
python test.py none_cnn5

# classify lbp3
python extract_features.py lbp3

python train.py lbp3 nn
python test.py lbp3_nn

python train.py lbp3 cnn1
python test.py lbp3_cnn1

python train.py lbp3 cnn3
python test.py lbp3_cnn3

python train.py lbp3 cnn5
python test.py lbp3_cnn5

# classify lbp5
python extract_features.py lbp5

python train.py lbp5 nn
python test.py lbp5_nn

python train.py lbp5 cnn1
python test.py lbp5_cnn1

python train.py lbp5 cnn3
python test.py lbp5_cnn3

python train.py lbp5 cnn5
python test.py lbp5_cnn5

# classify lbp7
python extract_features.py lbp7

python train.py lbp7 nn
python test.py lbp7_nn

python train.py lbp7 cnn1
python test.py lbp7_cnn1

python train.py lbp7 cnn3
python test.py lbp7_cnn3

python train.py lbp7 cnn5
python test.py lbp7_cnn5
