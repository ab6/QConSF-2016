'''
Name: Module2-DataAnalysis-Solution.py
Date: November 11, 2016
Author: Amber McKenzie
QCon San Francisco 2016
https://github.com/ab6/QConSF-2016.git

This module is designed to present NLP basics from the NLTK within a basic data analysis application.
It covers word tokenization, sentence tokenization, bigrams and collocations, part-of-speech (pos) tagging, and named entity tagging.
For additional information and prerequisites, see the readme on the github repo.

Notes:
-Code will take a while to run if run on the entire state-union corpus.
-Code will not proceed past the plots until each plot window is closed.
'''


import nltk
from nltk.collocations import *
from nltk.corpus import state_union
# import matplotlib.pyplot as plt

def extract_entity_names(t):
    '''
    Extract entity names from named entity tree returned by NLTK ne_chunk_sents method
    :param t: tree with pos and NE tags
    :return: list of named entities
    '''
    entity_names = []
    if hasattr(t, 'label'):
        if t.label() == 'NE':
            entity_names.append(' '.join([child[0] for child in t]), )
        else:
            for child in t:
                entity_names.extend(extract_entity_names(child))
    return entity_names

def extract_entities(taggedText):
    '''
    Create map with entity and their counts
    :param taggedText: Parsed text (output of ne chunker) in tree form
    :return: dict of entities and their freq counts
    '''
    entity_names = []
    for tree in taggedText:
        entity_names.extend(extract_entity_names(tree))
    return entity_names


#get year and words for each file
extracted= [(state_union.raw(fileid), int(fileid[:4])) for fileid in state_union.fileids()]
docs, years = zip(*extracted)

#break text down into sentences, tokens
tokens = [nltk.word_tokenize(text) for text in docs]
sents = [nltk.sent_tokenize(text.replace("\n", " ")) for text in docs]
senttokens = [[nltk.word_tokenize(sent) for sent in entry] for entry in sents]

# #get counts of unique words and plot over time
# unique = [len(set(words)) for words in tokens]
# plt.scatter(years, unique)
# plt.show()
#
# #get unique/total ratio
# ratios = [(float(len(set(words)))/float(len(words))) for words in tokens]
# plt.scatter(years, ratios)
# plt.show()

#Collocations
lower = [[word.lower() for word in words] for words in tokens]
bigram_measures = nltk.collocations.BigramAssocMeasures()
for i in range(len(years)):
    finder = BigramCollocationFinder.from_words(lower[i])
    # finder.apply_freq_filter(3)
    print (years[i], finder.nbest(bigram_measures.pmi, 10))

#chunk text and extract entities
postags = [nltk.pos_tag_sents(entry) for entry in senttokens]
ne_tags = [nltk.ne_chunk_sents(pos, binary=True) for pos in postags]
ents = [extract_entities(tagged) for tagged in ne_tags]
entFreqs = [nltk.FreqDist(entry) for entry in ents]

#get freq dist of all entities
allentities = [item for sublist in ents for item in sublist]
allentfreq = nltk.FreqDist(allentities)

#make list of top 50 most frequent and prune individual docs to take out filtered words
filtered, freq = zip(*allentfreq.most_common(50))
pruned = []
for entFreq in entFreqs:
    ents, freqs = zip(*entFreq.most_common(100))
    topEnts = [x for x in ents if x not in filtered]
    pruned.append(topEnts)

print ("Top 10 entities by year")
for i in range(len(years)):
    print (years[i], pruned[i][:10])



