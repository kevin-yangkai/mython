from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from datasets import load_imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D


np.random.seed(1337)  # for reproducibility
# set parameters:
max_features = 100
maxlen = 60
batch_size = 32
embedding_dims = 60
nb_filter = 10
filter_length = 3
hidden_dims = 20
nb_epoch = 3
print("Loading data...")
(X_train, y_train), (X_test, y_test) = load_imdb(imdb_dataset="datasets/6singers_13mfcc_nolink.pkl",
                                                 nb_words=max_features,
                                                 test_split=0.2)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')
print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('Build model...')
model = Sequential()
# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
model.add(Dropout(0.25))
# we add a Convolution1D, which will learn nb_filter
# word group filters of size filter_length:
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode="valid",
                        activation="relu",
                        subsample_length=1))
# we use standard max pooling (halving the output of the previous layer):
model.add(MaxPooling1D(pool_length=2))
# We flatten the output of the conv layer, so that we can add a vanilla dense layer:
model.add(Flatten())
# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.25))
model.add(Activation('relu'))
# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('softmax'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', class_mode="binary")
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True,
          validation_data=(X_test, y_test))
