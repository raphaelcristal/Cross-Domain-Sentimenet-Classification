# -*- coding:utf-8 -*-
from sqlite3 import dbapi2 as sqlite
from os import path
from sys import argv
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy import sparse
from sklearn.svm.sparse import LinearSVC
from bisect import bisect_left

from nltk.corpus import wordnet as wn
from nltk.corpus import brown
from nltk.corpus import stopwords
from nltk import BigramTagger
from nltk import UnigramTagger
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
        self._featuresWithSynsets = {}
        self._featuresWithoutSynsets = {}
        self._allSynsets = []

    def _getFeatures(self, maxDIFeatures=500, minFrequency=5):
        features = []
        self._cursor.execute("SELECT term FROM " +self._tableName+ " WHERE freqSource + freqTarget >= ?", [minFrequency])
        features = [a[0] for a in self._cursor.fetchall()]
        self._cursor.execute("SELECT term FROM mostinformatives") 
        mostInformatives = set([a[0] for a in self._cursor.fetchall()][30000:-30000])
        features = [feature for feature in features if feature not in mostInformatives]
        return sorted(features[:maxDIFeatures]), sorted(features[maxDIFeatures:])

    def _getSynsets(self, domainIndependentFeatures, minSyn):
	#unigramTagger = UnigramTagger(brown.tagged_sents(simplify_tags=True))
        #bigramTagger = BigramTagger(brown.tagged_sents(simplify_tags=True), backoff=unigramTagger)
	#taggedBigrams = [bigramTagger.tag(feature.split('_')) for feature in domainIndependentFeatures if "_" in feature and "<" not in feature]
        #tmp = ("PRO", "CNJ", "DET", "EX", "MOD", "P", "TO")
        #for x in taggedBigrams:
            #firstWord,firstTag = x[0]
            #secondWord,secondTag = x[1]
            #feature = "_".join((firstWord,secondWord))
            #if firstTag in tmp and secondTag not in tmp:
                #self._featuresWithSynsets[feature] = wn.synsets(secondWord)
            #elif firstTag not in tmp and secondTag in tmp:
                #self._featuresWithSynsets[feature] = wn.synsets(firstWord)


        Bigrams = [feature for feature in domainIndependentFeatures if "_" in feature and "<" not in feature]
        #filterWords = ("a", "and", "are", "be", "has", "have", "i", "is", "it", "of", "the", "to", "will", "had", "as", "my", "that", "was")
        stopwordList = set(stopwords.words("english")) - set(("no", "nor", "not"))
        for bigram in Bigrams:
            firstWord, secondWord = bigram.split("_")
            if firstWord in stopwordList and secondWord in stopwordList:
                pass
            elif firstWord in stopwordList:
                self._featuresWithSynsets[bigram] = wn.synsets(secondWord)
            elif secondWord in stopwordList:
                self._featuresWithSynsets[bigram] = wn.synsets(firstWord)

        self._featuresWithSynsets = {feature:[str(synset) for synset in synsets] for feature,synsets in self._featuresWithSynsets.items() if synsets}
        unigrams = [feature for feature in domainIndependentFeatures if "_" not in feature]
        for unigram in unigrams:
            synsets = wn.synsets(unigram)
            if synsets:
                self._featuresWithSynsets[unigram] = [str(synset) for synset in synsets]

        allSynsets = [synsets for sublist in self._featuresWithSynsets.values() for synsets in sublist]
        allSynsets = set([synset for synset in allSynsets if allSynsets.count(synset) >= minSyn])
        self._featuresWithSynsets = {feature:set(synsets) & allSynsets for feature,synsets in self._featuresWithSynsets.items() if set(synsets) & allSynsets}
        self._featuresWithoutSynsets = sorted(set(domainIndependentFeatures) - set(self._featuresWithSynsets.keys()))
        return sorted(allSynsets)

    def _createCooccurrenceMatrix(self, domainIndependentFeatures, domainDependentFeatures):
        domainIndependentFeaturesSet = set(domainIndependentFeatures)
        domainDependentFeaturesSet = set(domainDependentFeatures)
        numSyn = len(self._allSynsets)
        def __parseFile(filePath):
            with open(filePath, "r") as f:
                for review in f:
                        reviewFeatures = set([tupel.split(":")[0].decode("utf-8") for tupel in review.split()])
                        independentFeatures = reviewFeatures & domainIndependentFeaturesSet
                        dependentFeatures = reviewFeatures & domainDependentFeaturesSet
                        for dependentFeature in dependentFeatures:
                            rowIndex = bisect_left(domainDependentFeatures,dependentFeature)
                            for independentFeature in independentFeatures:
                                if independentFeature in self._featuresWithSynsets:
                                    for synset in self._featuresWithSynsets[independentFeature]:
                                        matrix[rowIndex, bisect_left(self._allSynsets,synset)] += 1
                                else:
                                    matrix[rowIndex, bisect_left(self._featuresWithoutSynsets,independentFeature)+numSyn] += 1
                        
        matrix = np.zeros((len(domainDependentFeatures), len(self._featuresWithoutSynsets)+numSyn))
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
        numSynsets = len(self._allSynsets)
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
                        if feature in self._featuresWithSynsets:
                            for synset in self._featuresWithSynsets[feature]:
                                domainIndepIndizes.append(bisect_left(self._allSynsets,synset))
                                domainIndepValues.append(1)
                        else:
                            domainIndepIndizes.append(bisect_left(self._featuresWithoutSynsets,feature)+numSynsets)
                            domainIndepValues.append(1)
                            #domainIndepValues.append(reviewDict[feature])
                    for feature in domainDepReviewFeatures:
                        #domainDepValues.append(reviewDict[feature])
                        domainDepValues.append(1)
                        domainDepIndizes.append(bisect_left(domainDependentFeatures,feature))
                    domainIndepVector = sparse.csr_matrix((domainIndepValues,(np.zeros(len(domainIndepIndizes)),domainIndepIndizes)),
                            shape=(1,len(self._featuresWithoutSynsets)+numSynsets))
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




    def go(self,K=100, Y=6, DI=500, minFreq=5, minSyn=10):
        print self._sourceDomain + " -> " + self._targetDomain
        domainIndependentFeatures, domainDependentFeatures = self._getFeatures(DI,minFreq)
        numDomainIndep = len(domainIndependentFeatures)
        numDomainDep = len(domainDependentFeatures)
        #print "number of independent features %i, number of dependent features %i" % (numDomainIndep, numDomainDep)
        #print "finding synsets..."
        self._allSynsets = self._getSynsets(domainIndependentFeatures, minSyn)
        print self._featuresWithSynsets
        for k,v in self._featuresWithSynsets.items():
            print str(k) + " : " + str(v)
        if not self._allSynsets:
            return
        #print "creating cooccurrenceMatrix..."
        a = self._createCooccurrenceMatrix(domainIndependentFeatures, domainDependentFeatures)
        #print "creating SquareAffinityMatrix..."
        a = self._createSquareAffinityMatrix(a)
        #print "creating DiagonalMatrix..."
        b = self._createDiagonalMatrix(a)
        #print "multiplying..." 
        c = b.dot(a)
        del a
        c = c.dot(b)
        del b
        #print "calculating eigenvalues and eigenvectors"
        eigenValues,eigenVectors = eigsh(c, k=K, which="LA")
        del c
        #print "building document vectors..."
        documentVectorsTraining,classifications = self._createDocumentVectors(domainDependentFeatures, domainIndependentFeatures,self._sourceDomain)
        documentVectorsTesting,classificatons = self._createDocumentVectors(domainDependentFeatures, domainIndependentFeatures,self._targetDomain)
        #print "training and testing..."
        U  = [eigenVectors[:,x].reshape(np.size(eigenVectors,0),1) for x in eigenValues.argsort()[::-1]]
        U = np.concatenate(U,axis=1)[:numDomainDep]
        U = sparse.csr_matrix(U)
        clustering = [vector[1].dot(U).dot(Y).astype(np.float64) for vector in documentVectorsTraining]
        trainingVectors = [sparse.hstack((documentVectorsTraining[x][0],documentVectorsTraining[x][1],clustering[x])) for x in range(np.size(documentVectorsTraining,axis=0))]
        self._trainClassifier(trainingVectors,classifications)
        clustering = [vector[1].dot(U).dot(Y).astype(np.float64) for vector in documentVectorsTesting]
        testVectors = [sparse.hstack((documentVectorsTesting[x][0],documentVectorsTesting[x][1],clustering[x])) for x in range(np.size(documentVectorsTesting,axis=0))]
        print "accuracy: %.2f with K=%i AND DI=%i AND Y=%.1f AND minFreq=%i AND minSyn=%i" % (self._testClassifier(testVectors,classifications)*100,K,DI,Y,minFreq,minSyn)

if __name__ == "__main__":
    source = argv[1]
    target = argv[2]
    syn = int(argv[3])
    sfa = SpectralFeatureAlignment("/home/raphael/BachelorThesis/DataBig","/home/raphael/BachelorThesis/Data/processed_acl", source, target)
    sfa.go(minSyn=syn);
