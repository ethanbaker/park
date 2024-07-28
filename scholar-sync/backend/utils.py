import sys
import re
import numpy as np
import os
from gensim.models import KeyedVectors

# Load the Word2Vec model
path = os.path.join(os.getcwd(), "model", "word2vec-google-news-300.gz")
model = KeyedVectors.load_word2vec_format(path, binary=True)

# Load the common words library
common_words = []
with open(os.path.join("model", "common.txt")) as f:
    common_words = [word.lower().strip() for word in f.readlines()]

# Turn a string into a vector representation based on the given model
def string_to_vector(str):
    # Tokenize the sentence
    tokens = re.sub(r"[^a-z ]+", "", str.lower()).split(" ")
    tokens = [token for token in tokens if not token in common_words]

    # Get the vector representation for each token
    token_vectors = []
    for token in tokens:
        if token in model:
            token_vectors.append(model[token])

    # Return the average of the token vector
    return np.array(np.mean(token_vectors, axis=0))

# Turn a number into a vector representation
def num_to_vector(num):
    return np.array([num])