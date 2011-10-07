# -*- coding:utf-8 -*-
from sqlite3 import dbapi2 as sqlite
from os import path
import numpy as np
class SpectralFeatureAlignment():

    def __init__(self, dbDir, rawDataFolder, sourceDomain, targetDomain):
        self._dbDir = dbDir
        self._sourceDomain = sourceDomain
        self._rawDataFolder = rawDataFolder
        self._targetDomain = targetDomain
        self._tableName = sourceDomain + "to" + targetDomain
        self._connection = sqlite.connect(path.join(dbDir,sourceDomain))
        self._cursor = self._connection.cursor()

    def _getFeatures(self, maxDIFeatures=500, minFrequency=5):
        features = []
        self._cursor.execute("SELECT term FROM bookstodvd WHERE freqSource + freqTarget >= ?", [minFrequency])
        features = [a[0] for a in self._cursor.fetchall()]
        return features[:maxDIFeatures], features[maxDIFeatures:]

    def _createCooccurrenceMatrix(self, domainIndependentFeatures, domainDependentFeatures):
        domainIndependentFeaturesSet = set(domainIndependentFeatures)
        domainDependentFeaturesSet = set(domainDependentFeatures)
        def __parseFile(filePath):
            with open(filePath, "r") as f:
                for review in f:
                        reviewFeatures = set([tupel.split(":")[0].decode("utf-8") for tupel in review.split()])
                        independentFeatures = reviewFeatures & domainIndependentFeaturesSet
                        dependentFeatures = reviewFeatures & domainDependentFeaturesSet
                        for dependentFeature in dependentFeatures:
                            rowIndex = domainDependentFeatures.index(dependentFeature)
                            for independentFeature in independentFeatures:
                                matrix[rowIndex, domainIndependentFeatures.index(independentFeature)] += 1
                        
        matrix = np.zeros((len(domainDependentFeatures), len(domainIndependentFeatures)))
        __parseFile(path.join(self._rawDataFolder, self._sourceDomain, "positive.review"))
        __parseFile(path.join(self._rawDataFolder, self._sourceDomain, "negative.review"))
        __parseFile(path.join(self._rawDataFolder, self._targetDomain, "positive.review"))
        __parseFile(path.join(self._rawDataFolder, self._targetDomain, "negative.review"))
        return matrix

    def _createSquareAffinityMatrix(self, cooccurrenceMatrix):
       height = np.size(cooccurrenceMatrix, 0) 
       width = np.size(cooccurrenceMatrix, 1) 
       topMatrix = np.zeros((height, height))
       topMatrix = np.concatenate((topMatrix, cooccurrenceMatrix), axis=1)
       bottomMatrix = np.zeros((width,width))
       bottomMatrix = np.concatenate((np.transpose(cooccurrenceMatrix), bottomMatrix), axis=1)
       matrix = np.concatenate((topMatrix, bottomMatrix), axis=0)
       return matrix
   
    def _createDiagonalMatrix(self, squareAffinityMatrix):
        matrix = np.zeros((np.size(squareAffinityMatrix,0),np.size(squareAffinityMatrix, 1)))
        tmp = []
        for i,x in enumerate(squareAffinityMatrix):
            rowSum = np.sum(x)
            if rowSum == 0:
                matrix[i][i] = 0     
                tmp.append(0)
            else:
                matrix[i][i] = np.sqrt(1.0 / rowSum)
                tmp.append(np.sqrt(1.0 / rowSum))
        np.save("myD", tmp)
        return matrix



    def go(self):
        domainIndependentFeatures, domainDependentFeatures = self._getFeatures(300,18)
        print "independent " + str(len(domainIndependentFeatures)) + " dependent " + str(len(domainDependentFeatures))
        print "creating cooccurrenceMatrix..."
        a = self._createCooccurrenceMatrix(domainIndependentFeatures, domainDependentFeatures)
        print "creating SquareAffinityMatrix..."
        a = self._createSquareAffinityMatrix(a)
        np.save("myA", a)
        print "creating DiagonalMatrix..."
        b = self._createDiagonalMatrix(a)
        print "multiplying..." 
        c = b.dot(a).dot(b)
        print np.size(c,axis=0),np.size(c,axis=1)
        eigenValues, eigenVectors = np.linalg.eig(c)
        print eigenValues[0]
        print eigenVectors[:,0]















sfa = SpectralFeatureAlignment("/home/raphael/BachelorThesis/Data","/home/raphael/BachelorThesis/Data/processed_acl", "books", "dvd")
sfa.go()



