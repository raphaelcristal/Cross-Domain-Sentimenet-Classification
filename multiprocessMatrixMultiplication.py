# -*- coding:utf-8 -*-

import numpy as np
from multiprocessing import Pool 

def _calculateBlock(blocks):
    return blocks[0].dot(blocks[1])

def matrixMultiplication(A,B):
    pool = Pool(processes=4)
    blockSize = np.size(A,0) / 2
    taskList = ((A[:blockSize,:],B[:,:blockSize]),
                (A[:blockSize,:],B[:,blockSize:]),
                (A[blockSize:,:],B[:,:blockSize]),
                (A[blockSize:,:],B[:,blockSize:]))
    results = pool.map(_calculateBlock,taskList)
    return np.concatenate((
            np.concatenate((results[0], results[1]), axis=1),
            np.concatenate((results[2], results[3]), axis=1))
                            ,axis=0)    
