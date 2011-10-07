# -*- coding:utf-8 -*-
from sys import argv
import math
from sqlite3 import dbapi2 as sqlite
from os import path
class feature:
    def __init__(self,word):
        self.word = word
        self.overallOccurrence = 0
        self.occurrenceByFrequency = {}
    
    def addOccurrences(self,tupel):
        occurrenceCount = tupel[1]
        self.occurrenceByFrequency.setdefault(occurrenceCount, 0)
        self.occurrenceByFrequency[occurrenceCount] += 1
        self.overallOccurrence += int(occurrenceCount) 

def _parseFile(filePath, features={}):
    reviewCount = 0
    with open(filePath, "r") as file:
        for review in file:
            tupelList = [tupel.split(":") for tupel in review.split() if
            "#label#" not in tupel]
            reviewCount += 1
            for tupel in tupelList:
                if(tupel[0].isalnum()):
                    features.setdefault(tupel[0], feature(tupel[0]))
                    features[tupel[0]].addOccurrences(tupel)

    return reviewCount,features

def _calculateMutualInformation(sourceFeatures, targetFeatures, overallCountSource, overallCountTarget):
    MutualInformation = {}
    allWords = set(sourceFeatures.keys()) | set(targetFeatures.keys())
    pdSource = float(overallCountSource)/float(overallCountSource + overallCountTarget)
    pdTarget = float(overallCountTarget)/float(overallCountSource + overallCountTarget)
    overallCountSum = float(overallCountTarget + overallCountSource)
    def _calc(sourceFeatures, targetFeatures, featureName, pd):
        mutualInformation = 0
        for occurrence,count in sourceFeatures[featureName].occurrenceByFrequency.items():
            pxd = 1.0 / overallCountSum * float(count) 
            if(targetFeatures.has_key(featureName) and targetFeatures[featureName].occurrenceByFrequency.has_key(occurrence)):
                px = 1.0 / overallCountSum * float((count + targetFeatures[featureName].occurrenceByFrequency[occurrence]))
            else:
                px = 1.0 / overallCountSum * float(count)
            mutualInformation += pxd *  math.log(pxd / (px * pd), 2) 
        #overallCount
        pxd = 1.0 / overallCountSum * float(sourceFeatures[featureName].overallOccurrence)
        if(targetFeatures.has_key(featureName)):
            px = 1.0 / overallCountSum * float(sourceFeatures[featureName].overallOccurrence + targetFeatures[featureName].overallOccurrence)
        else:
            px = 1.0 / overallCountSum * float(sourceFeatures[featureName].overallOccurrence)
        mutualInformation += pxd *  math.log(pxd / (px * pd), 2) 
        return mutualInformation
    
    for featureName in allWords:
        sourceMutualInformation = 0
        targetMutualInformation = 0
        if featureName in sourceFeatures:
            sourceMutualInformation = _calc(sourceFeatures,targetFeatures, featureName, pdSource) 
        if featureName in targetFeatures:
            targetMutualInformation = _calc(targetFeatures, sourceFeatures, featureName, pdTarget)
        
        MutualInformation[featureName] = sourceMutualInformation + targetMutualInformation

    return MutualInformation

def _MutualInformation(sourceDomain, targetDomain, rawDataFolder):
    sourceFeatures={}
    targetFeatures={}
    negativeReviewCountSource, sourceFeatures = _parseFile(path.join(rawDataFolder, sourceDomain, "negative.review"), sourceFeatures)
    positiveReviewCountSource, sourceFeatures = _parseFile(path.join(rawDataFolder, sourceDomain, "positive.review"), sourceFeatures)
    overallCountSource = negativeReviewCountSource + positiveReviewCountSource
    negativeReviewCountTarget, targetFeatures = _parseFile(path.join(rawDataFolder, targetDomain, "negative.review"), targetFeatures) 
    positiveReviewCountTarget, targetFeatures = _parseFile(path.join(rawDataFolder, targetDomain, "positive.review"), targetFeatures)
    overallCountTarget = negativeReviewCountTarget + positiveReviewCountTarget

    MutualInformation = _calculateMutualInformation(sourceFeatures,targetFeatures,overallCountSource,overallCountTarget)
    MIList = []
    for feature,mutualInformation in MutualInformation.items():
        sourceFeatureCount = sourceFeatures[feature].overallOccurrence if feature in sourceFeatures else 0
        targetFeatureCount = targetFeatures[feature].overallOccurrence if feature in targetFeatures else 0
        MIList.append((MutualInformation[feature], feature, sourceFeatureCount, targetFeatureCount))

    MIList.sort()
    return MIList

def _createCommonInformation(cursor, sourceDomain, rawDataFolder):
    featureCount = {}
    negativeReviewCount = 0
    positiveReviewCount = 0
    positiveWordCount = 0
    negativeWordCount = 0
    with open(path.join(rawDataFolder, sourceDomain, "positive.review")) as f:
        for review in f:
            tupelList = [tupel.split(":") for tupel in review.split() if "#label#" not in tupel]
            positiveReviewCount += 1
            positiveWordCount += len(tupelList)
            for tupel in tupelList:
                if(tupel[0].isalnum()):
                    featureCount.setdefault(tupel[0], [0,0])
                    featureCount[tupel[0]][0] += int(tupel[1])
    with open(path.join(rawDataFolder, sourceDomain, "negative.review")) as f:
        for review in f:
            tupelList = [tupel.split(":") for tupel in review.split() if "#label#" not in tupel]
            negativeReviewCount += 1 
            negativeWordCount += len(tupelList)
            for tupel in tupelList:
                if(tupel[0].isalnum()):
                    featureCount.setdefault(tupel[0], [0,0])
                    featureCount[tupel[0]][1] += int(tupel[1])

    mostInformatives = []
    for feature, count in featureCount.items():
        cursor.execute("INSERT INTO allterms VALUES (?,?,?)", (feature, count[1], count[0]))
        mostInformatives.append((count[1] - count[0], feature, count[1], count[0]))

    mostInformatives.sort(reverse=True)
    for entry in mostInformatives:
        cursor.execute("INSERT INTO mostInformatives VALUES (?,?,?,?)", (entry[1], entry[0], entry[2], entry[3]))

    cursor.execute("INSERT INTO summary VALUES (?,?)", ("number of positive reviews", positiveReviewCount))
    cursor.execute("INSERT INTO summary VALUES (?,?)", ("number of negative reviews", negativeReviewCount))
    cursor.execute("INSERT INTO summary VALUES (?,?)", ("words in positive reviews", positiveWordCount))
    cursor.execute("INSERT INTO summary VALUES (?,?)", ("words in negative reviews", negativeWordCount))
            


def _createTables(cursor):
    cursor.execute('''create table if not exists allterms (term text, freqBad int, freqGood int)''')
    cursor.execute('''create table if not exists summary (parameter text, Value int)''')
    cursor.execute('''create table if not exists mostInformatives (term text, diff int, freqBad int, freqGood int)''')


def main(sourceDomain, targetDomains, rawDataFolder, databaseFolder):
    connection = sqlite.connect(path.join(databaseFolder, sourceDomain))
    cursor = connection.cursor()
    _createTables(cursor)
    _createCommonInformation(cursor, sourceDomain, rawDataFolder)
    connection.commit()
    for targetDomain in targetDomains:
        MIList = _MutualInformation(sourceDomain, targetDomain, rawDataFolder)
        tableName = sourceDomain + "to" + targetDomain
        cursor.execute("create table if not exists "+ tableName +" (term text, MI real, FreqSource int, FreqTarget int)")
        for entry in MIList:
            cursor.execute("INSERT INTO "+ tableName +" VALUES (?,?,?,?)", (entry[1],entry[0],entry[2],entry[3]))
        connection.commit()
        termCount = cursor.execute("SELECT count(*) FROM "+ tableName).fetchone()[0]
        cursor.execute("INSERT INTO summary  VALUES (?,?)", (tableName, termCount))
        connection.commit()

    connection.close()


if __name__ == "__main__":
    try:
        sourceDomain = argv[1]
        targetDomains = argv[2].split(",")
        rawDataFolder = argv[3]
        databaseFolder = argv[4]
    except:
        print "USAGE: sourceDomain targetDomains(separated by comma) rawDataFolder databaseFolder"
        print "EXAMPLE: books electronics,dvd /rawData/processed_acl /databases"

    main(sourceDomain, targetDomains, rawDataFolder, databaseFolder) 

    


