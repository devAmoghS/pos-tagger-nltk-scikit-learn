# Split the dataset for training and testing
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from pos_tagger.corpus import tagged_sentences
from pos_tagger.utils import features, untag

cutoff = int(.75 * len(tagged_sentences))
training_sentences = tagged_sentences[:cutoff]
test_sentences = tagged_sentences[cutoff:]

print(len(training_sentences))
print(len(test_sentences))


def transform_to_dataset(tagged_sentences):
    """The classifer accepts features for single word,
    but our dataset is composed of sentences, so we
    perform untagging."""
    X, y = [], []

    for tagged in tagged_sentences:
        for index in range(len(tagged)):
            X.append(features(untag(tagged), index))
            y.append(tagged[index][1])

    return X, y


X, y = transform_to_dataset(training_sentences)

clf = Pipeline([
    ('vectorizer', DictVectorizer(sparse=False)),
    ('classifier', DecisionTreeClassifier(criterion='entropy'))
])

num_samples = 15000     # play with `num_samples` to tune your accuracy
clf.fit(X[:num_samples],
        y[:num_samples])

print('Training completed')     # Current Accuracy: 0.9053109432749478

X_test, y_test = transform_to_dataset(test_sentences)

print("Accuracy:", clf.score(X_test, y_test))

# save the model to disk
filename = 'dt_clf.sav'
pickle.dump(clf, open(filename, 'wb'))
