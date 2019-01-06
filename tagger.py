import pickle

from nltk import word_tokenize

from pos_tagger.utils import features


def pos_tag(sentence):
    """Loads the saved classifier
    and predicts the tags for given sentence"""
    clf = pickle.load(open('dt_clf.sav', 'rb'))
    tags = clf.predict([features(sentence, index) for index in range(len(sentence))])
    return list(zip(sentence, tags))


print(pos_tag(word_tokenize('This is my friend, Bruce.')))
# [('This', 'DT'), ('is', 'VBZ'), ('my', 'NN'), ('friend', 'NN'), (',', ','), ('Bruce', 'NNP'), ('.', '.')]
