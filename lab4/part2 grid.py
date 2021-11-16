import sklearn
from sklearn.datasets import load_files
import os

model_path = os.path.join('movie_reviews', '')
print(model_path)

moviedir = r'c:/Users/jakob/Documents/GitHub/tnm108/lab4/movie_reviews'

# loading all files. 
movie = load_files(moviedir, shuffle=True)
print(len(movie.data))
# target names ("classes") are automatically generated from subfolder names
print(movie.target_names)
categories = movie.target_names
# First file seems to be about a Schwarzenegger movie. 
print(movie.data[0][:500])

# Split data into training and test sets
from sklearn.model_selection import train_test_split
docs_train, docs_test, y_train, y_test = train_test_split(movie.data, movie.target, test_size = 0.20, random_state = 12)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
text_clf = Pipeline([
 ('vect', CountVectorizer()),
 ('tfidf', TfidfTransformer()),
 ('clf', MultinomialNB()),
])



text_clf.fit(docs_train, y_train)


import numpy as np
predicted = text_clf.predict(docs_test)
print("multinomialBC accuracy ",np.mean(predicted == y_test))


# training SVM classifier
from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([
 ('vect', CountVectorizer()),
 ('tfidf', TfidfTransformer()),
 ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42
,max_iter=5, tol=None)),
])
text_clf.fit(docs_train, y_train)
predicted = text_clf.predict(docs_test)
print("SVM accuracy ",np.mean(predicted == y_test))

from sklearn import metrics
print(metrics.classification_report(y_test, predicted,
 target_names=movie.target_names))


print(metrics.confusion_matrix(y_test, predicted))

from sklearn.model_selection import GridSearchCV
parameters = {
 'vect__ngram_range': [(1, 1), (1, 2)],
 'tfidf__use_idf': (True, False),
 'clf__alpha': (1e-2, 1e-3, 1e-4, 1e-5, 1e-6),
}

gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
gs_clf = gs_clf.fit(docs_train[:1600], y_train[:1600])

# Lol
# print(movie.target_names[gs_clf.predict(['God is love'])[0]])
print(gs_clf.best_score_)

for param_name in sorted(parameters.keys()):
 print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))