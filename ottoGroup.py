from __future__ import absolute_import
from __future__ import print_function

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

'''
    Tested with Python 2.7.9
    For GPU Utilization:
        Command: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python kaggle_otto_nn.py
    Data: https://www.kaggle.com/c/otto-group-product-classification-challenge/data
'''

randx = 1711                                                       ## random seed for reproducibility

initx = 'glorot_uniform'                                                   ## initialization function  for weighting individual units within each layer of the network
inity = 'glorot_uniform'
initz = 'glorot_uniform'
initxy = 'glorot_uniform'

hunit1 = 1024                                                       ## units in hidden layer 1
hunit2 = 1024
hunit3 = 1024
hunit4 = 1024

drop1 = 0.5                                                        ## dropout rate at hidden layer 1
drop2 = 0.5
drop3 = 0.5
drop4 = 0.5

optix = "adam"          ## optimization algorithm, "adam", keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True), "adagrad", "adadelta", or "rmsprop"
epx = 20                ## number of epochs
batchx = 256            ## batch size -         confirmed: by itself, minor effect on submission quality. can keep at 256 while testing.
v_split = 0.15          ## validation split
finact = 'softmax'      ## final activation to output

np.random.seed(randx) # for reproducibility

def load_data(path, train=True):
    df = pd.read_csv(path)
    X = df.values.copy()
    if train:
        np.random.shuffle(X) # https://youtu.be/uyUXoap67N8
        X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
        return X, labels
    else:
        X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
        return X, ids

def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder

def make_submission(y_prob, ids, encoder, fname):
    with open(fname, 'w') as f:
        f.write('id,')
        f.write(','.join([str(i) for i in encoder.classes_]))
        f.write('\n')
        for i, probs in zip(ids, y_prob):
            probas = ','.join([i] + [str(p) for p in probs.tolist()])
            f.write(probas)
            f.write('\n')
    print("Wrote submission to file {}.".format(fname))

print("Loading data...")
X, labels = load_data('data/train.csv', train=True)
X, scaler = preprocess_data(X)
y, encoder = preprocess_labels(labels)

X_test, ids = load_data('data/test.csv', train=False)
X_test, _ = preprocess_data(X_test, scaler)

nb_classes = y.shape[1]
print(nb_classes, 'classes')

dims = X.shape[1]
print(dims, 'dims')

print("Building model...")

model = Sequential()

# INPUT LAYER TO HIDDEN LAYER 1
model.add(Dense(dims, hunit1, init=initx))                 ## 'Base' of the layer; input and output shape as int >= 0, and declaration of initialization function for assigning weight to unit.
model.add(PReLU((hunit1,)))                                ## Activation Function
model.add(BatchNormalization((hunit1,)))                   ## Test swapping this out since it might be incorrectly utilized
model.add(Dropout(drop1))                                 ## Regularization - Set a percentage of input units to 0 at each update during training to prevent overfitting.

# HIDDEN LAYER 1 TO 2
model.add(Dense(hunit1, hunit2, init=inity))
model.add(PReLU((hunit2,)))
model.add(BatchNormalization((hunit2,)))                   ## Normalization of the outputs of the previous layer's activation functions.
model.add(Dropout(drop2))

# HIDDEN LAYER 2 TO 3
model.add(Dense(hunit2, hunit3, init=initz))
model.add(PReLU((hunit3,)))
model.add(BatchNormalization((hunit3,)))
model.add(Dropout(drop3))

# HIDDEN LAYER 3 TO 4
model.add(Dense(hunit3, hunit4, init=initz))
model.add(PReLU((hunit4,)))
model.add(BatchNormalization((hunit4,)))
model.add(Dropout(drop4))

# HIDDEN LAYER 4 TO OUTPUT LAYER
model.add(Dense(hunit4, nb_classes, init=initxy))
model.add(Activation(finact))                        ## Defined activation function for the last step in the network.

model.compile(loss='categorical_crossentropy', optimizer=optix)                ## Choose built-in optimization method.
# sgd = keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer=sgd)

print("Training model...")

model.fit(X, y, nb_epoch=epx, batch_size=batchx, validation_split=v_split)

print("Generating submission...")

proba = model.predict_proba(X_test)
make_submission(proba, ids, encoder, fname='keras-otto.csv')
