"""Given a network, generates text with it."""
import os
import sys

import numpy as np
from sklearn.externals import joblib

from model import make_model, prepare_text


def read_model(weight_filename, network):
    """Read weights and construct a network."""
    model = make_model(network.input_shape, network.output_shape)
    model.load_weights(weight_filename)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def generate_text(model, encoder):
    """Generate text with a model."""
    pattern = ('Above these apparent hieroglyphics was a figure of '
               'evidently pictorial intent, though its impression')
    pattern = encoder.encode(prepare_text(pattern))

    for i in range(1000):
        X = np.reshape(pattern, (1, len(pattern), 1))
        X = X / float(encoder.n_vocab)
        prediction = model.predict(X, verbose=0)
        best_answer = np.argmax(prediction)
        result = encoder.decode(best_answer)
        sys.stdout.write(result)
        sys.stdout.flush()
        pattern.append(best_answer)
        pattern = pattern[1:]

    print('\nDone.')


def generate():
    """Generate text for a given network model."""
    network = joblib.load('models/network.pkl')
    model = read_model(os.environ['WEIGHT_FILE'], network)
    encoder = joblib.load('models/encoder.pkl')
    generate_text(model, encoder)


generate()
