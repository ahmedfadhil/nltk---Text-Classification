import nltk
from nltk.corpus import movie_reviews
import random
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# documents = []
# for category in movie_reviews.categoes:
#     for fileid in movie_reviews.fileids(category):
#         documents.append(list(movie_reviews.words(fileid)), category)


random.shuffle(documents)

print(documents[1])

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

# print(all_words.most_common(23))
# print(all_words["whack"])

word_features = list(all_words.keys())[:3000]


def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
        return features


print((find_features(movie_reviews.words("neg/cv000_29416.txt"))))
featuresets = [(find_features(rev), category) for (rev, category) in documents]

training_set = featuresets[:1900]
test_set = featuresets[1900:]

classifier = nltk.NaiveBayesClassifier.train(training_set)
classifier_f = open("Naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

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
