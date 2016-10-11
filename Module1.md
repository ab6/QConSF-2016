#Workshop - Introduction to NLP using NLTK and Python
- QCon San Francisco 2016
- November 11, 2016
- Amber McKenzie

##General Notes:
Many helpful resources are available online including:
- The Python 3 Reference: https://docs.python.org/3/reference/
- The NLTK Book: http://www.nltk.org/book/
- The NLTK Module Reference: http://www.nltk.org/api/nltk.html
  - Specifically useful for Module 2: http://www.nltk.org/_modules/nltk/corpus/reader/plaintext.html
- NLTK Data: http://www.nltk.org/nltk_data/

##Module 1 - Document Categorization via Naive-Bayes and Decision Tree Classifier Models
In this module, we will explore document categorization based on word frequency distribution by using simple classifiers trained on a known corpus. We will then use a small subset of the data to test the accuracy of the models.  These techniques can be applied to other models and data sets.

The data for this application will be the Brown Corpus available via the Python NLTK package.  More information for each corpus in NLTK can be found on the NLTK.org website:
http://www.nltk.org/nltk_data/

To achieve this goal, do the following:
- Load the Brown Corpus documents word lists and categories
- Run a frequency distribution on all the words in the corpus
- Determine a set of words to be the feature set based on some threshold
- For each document, create a dict of each feature word that indicates if it exists in the document
- Pair the feature set (dict) for each document with the document category
- Shuffle the feature word-category pairings and split into two sets:
  - 80% into a training set
  - 20% into a test set
- Run the training set through a Naive-Bayes Classifier
- Run an accuracy test on the classifier using the created Naive-Bayes Classifier and the test set
- Print the resulting accuracy
- Print out most informative features (can only do this for the NB classifier)
- Try the same approach with a Decision Tree classifier

###Module 1 Hints and Tricks
- The Brown Corpus is a CategorizedTaggedCorpusReader
- Words for the corpus can be accessed via the .words() function
- Passing an array of file ids will return all matching words
- Categories can be accessed via the .categories() function
- Passing an array of file ids will return all matching categories
- The Naive-Bayes Classifier is a simple classification algorithm
  - Available through NLTK: http://www.nltk.org/_modules/nltk/classify/naivebayes.html
- Use a smaller number of features while debugging. Once debugged, you can up the number of features.
- Try the same approach with a Decision Tree classifier.
  - Available through NLTK: http://www.nltk.org/_modules/nltk/classify/decisiontree.html

###Module 1 Walkthrough
- Load the Brown corpus data into word list/category pairs
- Load all words from the Brown corpus
- Run a frequency distribution on all the words to order words by number of appearances
- Chop off the ordered word list at an arbitrary number or find threshold in the frequency count that scales down the size of the word list to a manageable feature word set
- For each document, create a dict of every feature word with the value a boolean for the existence of the word in the document
- Create a list of tuples of the document feature word dicts and the document category
- Shuffle the tuple list using random.shuffle
- Break the shuffled tuple list into two lists: 
  - 20% into test 
  - 80% into training
- Use the NLTK Naive Bayes Classifier on the training set to generate a model
- Calculate the model accuracy by passing in the classifier and the test set into NLTK Classifier Analyze
- Print the calculated accuracy on the test set
- Use the show_most_informative_features function to print out significant features for the naive-bayes model
- Try the same approach with a Decision Tree classifier.

###Module 1 Additional Goals
- Try filtering the feature word list using stopwords.  
  - How does this affect the accuracy?
- Use the output of most_informative_features to filter your feature list.
- Use a train, test and validation set to tune the model. A good description of how to do this is given in Section 1.2 of Chapter 6 of the NLTK book.
- Break the Brown corpus documents into into imaginative and informative prose lists and run a model on each list in a similar fashion.
  - Is the accuracy any better for the separated document lists?

