"""Script to build a LSTM Neural Network for text generation"""
import os

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from sklearn.externals import joblib

from model import make_model, prepare_text, Network
from encoder import TextEncoder


def read_input(filename):
    """Read an input file and return it prepared for enconding."""
    raw_text = open(filename).read()
    return prepare_text(raw_text)


def build_dataset(text, encoder, window_size=100):
    """Create a dataset (X, y) with a sliding window over the text."""
    n_chars = len(text)
    X, y = list(), list()
    # Slide window of |window_size| from the beginning to the end.
    for i in range(0, n_chars - window_size, 1):
        input_data = text[i:i + window_size]
        output_expected = text[i + window_size]
        X.append(encoder.encode(input_data))
        y.append(encoder.encode(output_expected))
    n_patterns = len(X)
    print(f'Patterns  : {n_patterns:,d}')
    X = np.reshape(X, (n_patterns, window_size, 1))
    X = X / float(encoder.n_vocab)
    y = np_utils.to_categorical(y)
    return (X, y)


def make_checkpoint():
    """Create a callback for saving checkpoints."""
    filepath = 'models/weights-{epoch:02d}-{loss:.4f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1,
                                 save_best_only=True, mode='min')
    return [checkpoint]


def train():
    """Read dataset and build a LSTM model."""
    # Read data and prepare a dataset.
    text = read_input(os.environ['INPUT_TEXT'])
    encoder = TextEncoder(text)
    joblib.dump(encoder, 'models/encoder.pkl')
    X, y = build_dataset(text, encoder)
    # Build model.
    network = Network(input_shape=(X.shape[1], X.shape[2]),
                      output_shape=y.shape[1])
    joblib.dump(network, 'models/network.pkl')
    model = make_model(network)
    model.fit(X, y, epochs=50, batch_size=128, callbacks=make_checkpoint())


train()
