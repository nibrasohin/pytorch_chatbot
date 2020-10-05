import nltk
import numpy as np
nltk.download('punkt')

from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    sentence_words = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for i, w in enumerate(all_words):
        if w in sentence_words:
            bag[i] = 1
    return bag

# tokenized_sentence = ['Hi', 'how', 'are', 'you']
# all_words = ['hi', 'lol', 'how', 'is', 'are', 'you']

# bags = bag_of_words(tokenized_sentence, all_words)
# print(bags)
