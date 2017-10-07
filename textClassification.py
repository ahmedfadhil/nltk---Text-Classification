import nltk
from nltk.corpus import movie_reviews
import random
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifier

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


# documents = [(list(movie_reviews.words(fileid)), category)
#              for category in movie_reviews.categories()
#              for fileid in movie_reviews.fileids(category)]

# documents = []
# for category in movie_reviews.categoes:
#     for fileid in movie_reviews.fileids(category):
#         documents.append(list(movie_reviews.words(fileid)), category)

# We commented it out to test for the positive and negative
# random.shuffle(documents)

# print(documents[1])
#
# all_words = []
#
# for w in movie_reviews.words():
#     all_words.append(w.lower())

short_pos = open("Sentiment Data/positive.txt", "r").read()
short_neg = open("Sentiment Data/negative.txt", "r").read()

documents = []
for r in short_pos('\n'):
    documents.append((r, "pos"))

for r in short_neg('\n'):
    documents.append((r, "neg"))

all_words = []

short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

for w in short_pos_words:
    all_words.append(w.lower())

for w in short_neg_words:
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

# print(all_words.most_common(23))
# print(all_words["whack"])

word_features = list(all_words.keys())[:5000]


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
        return features


print((find_features(movie_reviews.words("neg/cv000_29416.txt"))))
featuresets = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featuresets)
# Positive
training_set = featuresets[:10000]
test_set = featuresets[10000:]

# # Negative
# training_set = featuresets[100:]
# test_set = featuresets[:100]

classifier = nltk.NaiveBayesClassifier.train(training_set)
# classifier_f = open("Naivebayes.pickle", "rb")
# classifier = pickle.load(classifier_f)
# classifier_f.close()

print("NaiveBayes  Algo accuracy percent:", (nltk.classify.accuracy(classifier, test_set)) * 100)

classifier.show_most_informative_features(15)

# save_classifier = open("Naivebayes.pickle", "wb")
#
# pickle.dump(classifier, save_classifier)
# save_classifier.close()

# MultinomialNB
MNB_classifier = SklearnClassifier(MultinomialNB)
MNB_classifier.train(training_set)
print("MNB Classifier Algo accuracy percent:", (nltk.classify.accuracy(MNB_classifier, test_set)) * 100)

# BernoulliNB
BernoulliNB_classifier = SklearnClassifier(BernoulliNB)
BernoulliNB_classifier.train(training_set)
print("BernoulliNB Classifier Algo accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, test_set)) * 100)

# GaussianNB
GaussianNB_classifier = SklearnClassifier(GaussianNB)
GaussianNB_classifier.train(training_set)
print("GaussianNB Classifier Algo accuracy percent:", (nltk.classify.accuracy(GaussianNB_classifier, test_set)) * 100)

# LogisticRegression
LogisticRegression_classifier = SklearnClassifier(LogisticRegression)
LogisticRegression_classifier.train(training_set)
print("LogisticRegression Classifier Algo accuracy percent:",
      (nltk.classify.accuracy(LogisticRegression_classifier, test_set)) * 100)

# SGDClassifier
SGDClassifier_classifier = SklearnClassifier(SGDClassifier)
SGDClassifier_classifier.train(training_set)
print("SGDClassifier Classifier Algo accuracy percent:",
      (nltk.classify.accuracy(SGDClassifier_classifier, test_set)) * 100)

# SVC
SVC_classifier = SklearnClassifier(SVC)
SVC_classifier.train(training_set)
print("SVC Classifier Algo accuracy percent:", (nltk.classify.accuracy(SVC_classifier, test_set)) * 100)

# LinearSVC
LinearSVC_classifier = SklearnClassifier(LinearSVC)
LinearSVC_classifier.train(training_set)
print("LinearSVC Classifier Algo accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, test_set)) * 100)

# NuSVC
NuSVC_classifier = SklearnClassifier(NuSVC)
NuSVC_classifier.train(training_set)
print("NuSVC Classifier Algo accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, test_set)) * 100)

# Classifier count
voted_classifier = VoteClassifier(classifier, NuSVC, LinearSVC, LogisticRegression, SVC, SGDClassifier,
                                  GaussianNB_classifier, BernoulliNB_classifier, MNB_classifier)
print("Voted Classifier Algo accuracy percent:", (nltk.classify.accuracy(voted_classifier, test_set)) * 100)
print("Classification:", voted_classifier.classify(test_set[0][0]), "Confidence %:",
      voted_classifier.confidence(test_set[0][0]) * 100)
print("Classification:", voted_classifier.classify(test_set[1][0]), "Confidence %:",
      voted_classifier.confidence(test_set[1][0]) * 100)
print("Classification:", voted_classifier.classify(test_set[2][0]), "Confidence %:",
      voted_classifier.confidence(test_set[2][0]) * 100)
print("Classification:", voted_classifier.classify(test_set[3][0]), "Confidence %:",
      voted_classifier.confidence(test_set[3][0]) * 100)
print("Classification:", voted_classifier.classify(test_set[4][0]), "Confidence %:",
      voted_classifier.confidence(test_set[4][0]) * 100)
print("Classification:", voted_classifier.classify(test_set[5][0]), "Confidence %:",
      voted_classifier.confidence(test_set[5][0]) * 100)
