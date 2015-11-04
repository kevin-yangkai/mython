__author__ = 'zhangxulong'
# use pca for dimension reduction
from numpy import *


def pca(mata, length):
    meanVal = mean(mata, axis=0)
    stdVal = std(mata)
    rmmeanMat = (mata - meanVal) / stdVal
    covMat = cov(rmmeanMat, rowvar=0)
    eigval, eigvec = linalg.eig(covMat)
    maxnum = argsort(-eigval, axis=0)  # sort descend
    tfMat = eigvec[:, maxnum[0:length]]  # top length
    finalData = dot(rmmeanMat, tfMat)  #
    recoMat = finalData * tfMat.T * stdVal + meanVal
    return finalData, recoMat
