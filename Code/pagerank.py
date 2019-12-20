from __future__ import division

import numpy as np
from numpy.linalg import norm
from scipy.sparse import spdiags


def personalizedPageRank(imgImgArr, startVectors, c=0.15, allowedDiff=1e-9, maxIters = 100):
    # convert to (sparse) adjacency matrix
    sparseImgImgArray = imgImgArr

    # row normalize adjacency matrix
    m, n = sparseImgImgArray.shape
    d = sparseImgImgArray.sum(axis=1)

    # handling 0 entries
    d = np.maximum(d, np.ones((n, 1)))
    invd = 1.0 / d
    invd = np.reshape(invd, (1,-1))
    invD = spdiags(invd, 0, m, n)

    # row normalized adj. mat. 
    rowNormSparseImgImgArray = invD * sparseImgImgArray 
    rowNormSparseImgImgArrayTranspose = rowNormSparseImgImgArray.T
    
    # init seed vectors
    seedVectors = startVectors
    
    # init PPR
    current_ppr = seedVectors
    old_ppr = seedVectors
    diff = np.zeros((maxIters, 1))

    # iterate
    for i in range(maxIters):
        current_ppr = (1-c)*(rowNormSparseImgImgArrayTranspose.dot(old_ppr)) + c*seedVectors

        diff[i] = norm(current_ppr - old_ppr, 1)
        if diff[i] <= allowedDiff:
            break

        old_ppr = current_ppr
    
    # find and return top k
    return current_ppr