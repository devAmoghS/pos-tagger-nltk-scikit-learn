# Corpus for training

import nltk
# In case you are running for the first time, uncomment the below line and download the trrebank
# nltk.download('treebank')

tagged_sentences = nltk.corpus.treebank.tagged_sents()

print(tagged_sentences[0])
print("Tagged sentences: ", len(tagged_sentences))
print("Tagged words: ", len(nltk.corpus.treebank.tagged_words()))
