import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    pe = np.zeros((seq_len, d_model))
    position = np.arange(0, seq_len, dtype=np.float32).reshape(seq_len,1)
    div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * -(np.log(base)) / d_model)

    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)[:, : pe[:, 1::2].shape[1]]

    return pe