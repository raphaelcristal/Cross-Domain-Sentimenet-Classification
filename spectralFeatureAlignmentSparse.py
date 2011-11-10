# -*- coding:utf-8 -*-
from sqlite3 import dbapi2 as sqlite
from os import path
from sys import argv
import numpy as np
from scipy.sparse.linalg import eigs
from scipy import sparse
from sklearn.svm.sparse import LinearSVC
from bisect import bisect_left
class SpectralFeatureAlignment():

    def __init__(self, dbDir, rawDataFolder, sourceDomain, targetDomain):
        self._dbDir = dbDir
        self._sourceDomain = sourceDomain
        self._rawDataFolder = rawDataFolder
        self._targetDomain = targetDomain
        self._tableName = sourceDomain + "to" + targetDomain
        self._connection = sqlite.connect(path.join(dbDir,sourceDomain))
        self._cursor = self._connection.cursor()
        self._lsvc = LinearSVC(C=10000)

    def _getFeatures(self, maxDIFeatures=500, minFrequency=5):
        features = []
        self._cursor.execute("SELECT term FROM " +self._tableName+ " WHERE freqSource + freqTarget >= ?", [minFrequency])
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
                            #rowIndex = domainDependentFeatures.index(dependentFeature)
                            rowIndex = bisect_left(domainDependentFeatures,dependentFeature)
                            for independentFeature in independentFeatures:
                                matrix[rowIndex, bisect_left(domainIndependentFeatures,independentFeature)] += 1
                                #matrix[rowIndex, domainIndependentFeatures.index(independentFeature)] += 1
                        
        matrix = np.zeros((len(domainDependentFeatures), len(domainIndependentFeatures)))
        __parseFile(path.join(self._rawDataFolder, self._sourceDomain, "positive.review"))
        __parseFile(path.join(self._rawDataFolder, self._sourceDomain, "negative.review"))
        __parseFile(path.join(self._rawDataFolder, self._targetDomain, "positive.review"))
        __parseFile(path.join(self._rawDataFolder, self._targetDomain, "negative.review"))
        return sparse.coo_matrix(matrix)

    def _createSquareAffinityMatrix(self, cooccurrenceMatrix):
       height = np.size(cooccurrenceMatrix, 0) 
       width = np.size(cooccurrenceMatrix, 1) 
       topMatrix = sparse.coo_matrix((height,height))
       topMatrix = sparse.hstack((topMatrix,cooccurrenceMatrix))
       bottomMatrix = sparse.coo_matrix((width,width))
       bottomMatrix = sparse.hstack((cooccurrenceMatrix.transpose(), bottomMatrix))
       matrix = sparse.vstack((topMatrix, bottomMatrix))
       return matrix
   
    def _createDiagonalMatrix(self, squareAffinityMatrix):
        rows = range(squareAffinityMatrix.get_shape()[0])
        data = [0. if rowSum == 0 else np.sqrt(1.0 / rowSum) for rowSum in np.array(squareAffinityMatrix.sum(1)).reshape(-1,)]
        return sparse.coo_matrix((data,(rows,rows)),shape=(squareAffinityMatrix.get_shape()[0],squareAffinityMatrix.get_shape()[1]))

    def _createDocumentVectors(self,domainDependentFeatures, domainIndependentFeatures, domain):
        numDomainDep = len(domainDependentFeatures)
        numDomainIndep = len(domainIndependentFeatures)
        domainDepSet = set(domainDependentFeatures)
        domainIndepSet = set(domainIndependentFeatures)
        documentVectors = []
        classifications = []
        def __parseFile(filePath):
            with open(filePath,"r") as f:
                for review in f:
                    classification = 1 if "#label#:positive" in review else -1
                    reviewList = [tupel.split(":") for tupel in review.split() if "#label#" not in tupel]
                    reviewDict = {x[0].decode("utf-8"):int(x[1]) for x in reviewList}
                    reviewFeatures = set(reviewDict.keys())
                    domainDepReviewFeatures = domainDepSet & reviewFeatures
                    domainIndepReviewFeatures = domainIndepSet & reviewFeatures
                    domainDepValues,domainDepIndizes = [],[]
                    domainIndepValues, domainIndepIndizes = [],[]
                    for feature in domainIndepReviewFeatures:
                        domainIndepValues.append(reviewDict[feature])
                        #domainIndepIndizes.append(domainIndependentFeatures.index(feature))
                        domainIndepIndizes.append(bisect_left(domainIndependentFeatures,feature))
                    for feature in domainDepReviewFeatures:
                        domainDepValues.append(reviewDict[feature])
                        #domainDepIndizes.append(domainDependentFeatures.index(feature))
                        domainDepIndizes.append(bisect_left(domainDependentFeatures,feature))
                    domainIndepVector = sparse.csr_matrix((domainIndepValues,(np.zeros(len(domainIndepIndizes)),domainIndepIndizes)),shape=(1,numDomainIndep))
                    domainDepVector = sparse.csr_matrix((domainDepValues,(np.zeros(len(domainDepIndizes)),domainDepIndizes)),shape=(1,numDomainDep))
                    documentVectors.append((domainIndepVector,domainDepVector))
                    classifications.append(classification)

        __parseFile(path.join(self._rawDataFolder, domain, "positive.review"))
        __parseFile(path.join(self._rawDataFolder, domain, "negative.review"))
        return documentVectors,classifications 

    def _trainClassifier(self, trainingVectors, classifications):
        self._lsvc.fit(sparse.vstack(trainingVectors),classifications)

    def _testClassifier(self,testVectors,classifications):
        return self._lsvc.score(sparse.vstack(testVectors),classifications)




    def go(self,K=100, Y=0.6, DI=500, FREQ=1):
        print self._sourceDomain + " -> " + self._targetDomain
        domainIndependentFeatures, domainDependentFeatures = self._getFeatures(DI,FREQ)
        domainIndependentFeatures.sort()
        domainDependentFeatures.sort()
        numDomainIndep = len(domainIndependentFeatures)
        numDomainDep = len(domainDependentFeatures)
        print "number of independent " + str(numDomainIndep) + " number of dependent " + str(numDomainDep)
        print "creating cooccurrenceMatrix..."
        a = self._createCooccurrenceMatrix(domainIndependentFeatures, domainDependentFeatures)
        print "creating SquareAffinityMatrix..."
        a = self._createSquareAffinityMatrix(a)
        print "creating DiagonalMatrix..."
        b = self._createDiagonalMatrix(a)
        print "multiplying..." 
        c = b.dot(a)
        del a
        c = c.dot(b)
        del b
        print "calculating eigenvalues and eigenvectors"
        eigenValues,eigenVectors = eigs(c, k=K, which="LR")
        del c
        print "building document vectors..."
        documentVectorsTraining,classifications = self._createDocumentVectors(domainDependentFeatures, domainIndependentFeatures,self._sourceDomain)
        documentVectorsTesting,classificatons = self._createDocumentVectors(domainDependentFeatures, domainIndependentFeatures,self._targetDomain)
        print "training and testing..."
        self._lsvc = LinearSVC(C=10000)
        U  = [eigenVectors[:,x].reshape(np.size(eigenVectors,0),1) for x in eigenValues.argsort()]
        U = np.concatenate(U,axis=1)[:numDomainDep]
        clustering = [vector[1].dot(U).dot(Y).astype(np.float64) for vector in documentVectorsTraining]
        trainingVectors = [sparse.hstack((documentVectorsTraining[x][0],documentVectorsTraining[x][1],clustering[x])) for x in range(np.size(documentVectorsTraining,axis=0))]
        self._trainClassifier(trainingVectors,classifications)
        clustering = [vector[1].dot(U).dot(Y).astype(np.float64) for vector in documentVectorsTesting]
        testVectors = [sparse.hstack((documentVectorsTesting[x][0],documentVectorsTesting[x][1],clustering[x])) for x in range(np.size(documentVectorsTesting,axis=0))]
        print "accuracy: %f with K=%i AND DI=%i AND Y=%f" % (self._testClassifier(testVectors,classifications),K,DI,Y)

if __name__ == "__main__":
    source = argv[1]
    target = argv[2]
    sfa = SpectralFeatureAlignment("/home/raphael/BachelorThesis/DataBig","/home/raphael/BachelorThesis/Data/processed_acl", source, target)
    sfa.go();
