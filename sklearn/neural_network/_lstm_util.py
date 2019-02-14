import numpy as np

def cross_entropy(Y, Ytrue):
    denom = -np.log(Ytrue)
    grad = Y / denom
    return denom


def convert_to_one_hot_encoding(Y):
    new = []
    for cell in Y:
        a = np.zeros_like(cell)
        a[np.arange(len(cell)), cell.argmax(1)] = 1
        new.append(a)
    return np.array(new)

ERROR = {'cross_entropy':cross_entropy}
CONVERSION = {'convert_to_one_hot_encoding':convert_to_one_hot_encoding}

