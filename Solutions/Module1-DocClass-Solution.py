'''
Name: Module1-DocClass-Solution.py
Date: November 11, 2016
Author: Amber McKenzie
QCon San Francisco 2016
https://github.com/ab6/QConSF-2016.git

This module is designed to present an introduction to NLTK and machine learning basics in a document classification application.
For additional information and prerequisites, see the readme on the github repo.

Notes:
-Uncomment out the stopwords filter line to filter out stopwords.
'''


import nltk
import random
from nltk.corpus import brown
from nltk.corpus import stopwords


#get words with a specific freq count to where the total number of words is less than some threshold
def getFeatureWords(maxWordCount, words):
    featurewords = words
    freqThreshold = 5
    while len(featurewords) > maxWordCount:
        featurewords = [word for word in featurewords if words[word] > freqThreshold]
        freqThreshold += 1
    return featurewords

#create dict with boolean values for existence of words in a document
def getDocFeatures(doc, words):
    features = {}
    for word in words:
        features[word] = (word in doc)
    return features

#import data into words, category pairs
docs = [(list(brown.words(fileid)), category) for category in brown.categories() for fileid in brown.fileids(category)]

#identify list of words to be used as features
allwords = nltk.FreqDist([word.lower() for word in brown.words()])
featurewords = getFeatureWords(5000, allwords)

#filter for stopwords
#featurewords = [word for word in featurewords if word not in set(stopwords.words('english'))]

#create category, featureset pairs
docfeatures = [(getDocFeatures(doc, featurewords), category) for (doc, category) in docs]

#Break into training and test sets
random.shuffle(docfeatures)
train, test = docfeatures[:400], docfeatures[400:]

#train and test model
classifier = nltk.NaiveBayesClassifier.train(train)
accuracy = nltk.classify.accuracy(classifier, test)
print (accuracy)
print (classifier.show_most_informative_features(100))

classifier = nltk.classify.DecisionTreeClassifier.train(train)
accuracy = nltk.classify.accuracy(classifier, test)
print (accuracy)