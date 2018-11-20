"""Model preprocessing and architecture."""
from collections import namedtuple

from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential


Network = namedtuple('Network', ['input_shape', 'output_shape'])


def make_model(network):
    """Return a LSTM model."""
    model = Sequential()
    model.add(LSTM(256, input_shape=network.input_shape,
                   return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(network.output_shape, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def prepare_text(text):
    """Text preprocessing."""
    return text
