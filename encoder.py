"""Encoder and Decoder of texts for a LSTM model."""


class TextEncoder:
    """Encodes and decodes text for network input/output."""

    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.n_chars = len(text)
        self.n_vocab = len(chars)
        self._to_int = dict((c, i) for i, c in enumerate(chars))
        self._to_char = dict((i, c) for i, c in enumerate(chars))

    def encode(self, data):
        """
        Encode a single char or a string to integers.

        Parameters
        ----------
        data : str
            Either a single byte or a string.

        Returns
        -------
        int or list of ints
            - An int when the input is a single-char string.
            - A list of ints when the input is not a single-char string.
        """
        if len(data) == 1:
            self._to_int[data]
        return [self._to_int[char] for char in data]

    def decode(self, data):
        """
        Decode a single int or a list of integers to chars.

        Parameters
        ----------
        data : str
            Either a single int or a list of integers.

        Returns
        -------
        chr or list of chr
            - A chr when the input is a single integer.
            - A list of chr when the input is a list of integers.
        """
        if not isinstance(data, list):
            return self._to_char[data]
        return [self._to_char[point] for point in data]

