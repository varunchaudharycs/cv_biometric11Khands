import numpy as np


def cosine_similarity(x, y):
    x = np.array(x)
    y = np.array(y)
    return (np.dot(x, y))*1.0/((np.sum(x*x)*np.sum(y*y))**0.5)


def dot_pdt_similarity(x, y):
    x = np.array(x)
    y = np.array(y)
    return (np.dot(x, y))*1.0