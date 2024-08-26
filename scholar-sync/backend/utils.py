import re
import numpy as np
import os
from gensim.models import KeyedVectors

base_path = "scholar-sync/backend"
#base_path = ""

# Load the Word2Vec model
path = os.path.join(os.getcwd(), base_path, "model", "word2vec-google-news-300.gz")
model = KeyedVectors.load_word2vec_format(path, binary=True)

# Load the common words library
common_words = []
with open(os.path.join(base_path, "model", "common.txt")) as f:
    common_words = [word.lower().strip() for word in f.readlines()]

# Turn a string into a vector representation based on the given model
def string_to_vector(str) -> np.ndarray[any]:
    if str == None or str == "":
        return np.array([])

    # Tokenize the sentence
    tokens = re.sub(r" +", "", re.sub(r"[^a-z ]+", " ", str.lower())).split(" ")
    tokens = [token for token in tokens if not token in common_words]

    # Get the vector representation for each token
    token_vectors = []
    for token in tokens:
        if token in model:
            token_vectors.append(model[token])

    # If no tokens were found, return an empty array
    if len(token_vectors) == 0:
        return np.array([])

    # Return the average of the token vector
    return np.array(np.mean(token_vectors, axis=0))

# Turn a number into a vector representation
def num_to_vector(num):
    if num == None:
        return np.array([])

    return np.array([num])

# Turn an enum into a vector representation (the value of the enum is in index 0 and the size of the enum is in index 1)
def enum_to_vector(enum):
    if enum == None or enum.value == None:
        return np.array([])

    return np.array([enum.value])

"""
# Find the distance between two vectors
def distance(a, b):
    if len(a) != len(b):
        raise Exception("vectors a and b are not the same length")

    return np.linalg.norm(np.array(a) - np.array(b))

# Find the inverse distance between two vectors
def inverse_distance(a, b):
    return 1 - distance(a, b)

# Return 0 if vectors don't match and 1 if they do
def match(a, b):
    if len(a) != len(b):
        print(a, b)
        raise Exception("vectors a and b are not the same length")

    for i in range(len(a)):
        if a[i] != b[i]:
            return 0

    return 1

# Return 1 if vectors don't match and 1 if they do
def no_match(a, b):
    return 1 - match(a, b)
"""