import numpy as np


def euclidean_distance(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.sqrt(np.sum((v1-v2*1.0)**2))
    # v1 and v2 vector length is the same
#         sq_sum = 0
#         for i in range(0, len(v1)):
#             sq_sum = sq_sum + ((v2[i] - v1[i]) * (v2[i] - v1[i]))
#         return sqrt(sq_sum)


def chi_squared_distance(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    filt = np.logical_and(x != 0, y != 0)
    return 0.5*np.sum(1.0*((x[filt]-y[filt])**2)/(x[filt]+y[filt]))
