# All Import Statements Defined Here
# Note: Do not add to this list.
# All the dependencies you need, can be installed by running .
# ----------------

import sys
assert sys.version_info[0]==3
assert sys.version_info[1] >= 5

from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import pprint
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5]
import nltk
# nltk.download('reuters')
from nltk.corpus import reuters
import numpy as np
import random
import scipy as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

START_TOKEN = '<START>'
END_TOKEN = '<END>'

np.random.seed(0)
random.seed(0)
# ----------------

def read_corpus(category="crude"):
    """ Read files from the specified Reuter's category.
        Params:
            category (string): category name
        Return:
            list of lists, with words from each of the processed files
    """
    files = reuters.fileids(category)
    return [[START_TOKEN] + [w.lower() for w in list(reuters.words(f))] + [END_TOKEN] for f in files]

# reuters_corpus = read_corpus()
# pprint.pprint(reuters_corpus[:3], compact=True, width=100)

""" Question 1.1: Implement `distinct_words` [code] (2 points)

    Write a method to work out the distinct words (word types) that occur in the corpus. 
    You can do this with `for` loops, but it's more efficient to do it with Python list 
    comprehensions. In particular, [this](https://coderwall.com/p/rcmaea/flatten-a-list-of-lists-in-one-line-in-python) 
    may be useful to flatten a list of lists. 
    If you're not familiar with Python list comprehensions in general, here's more information](https://python-3-patterns-idioms-test.readthedocs.io/en/latest/Comprehensions.html).
    
    You may find it useful to use [Python sets]
    (https://www.w3schools.com/python/python_sets.asp) to remove duplicate words.
"""

def distinct_words(corpus):
    """ Determine a list of distinct words for the corpus.
        Params:
            corpus (list of list of strings): corpus of documents
        Return:
            corpus_words (list of strings): list of distinct words across the corpus, sorted (using python 'sorted' function)
            num_corpus_words (integer): number of distinct words across the corpus
    """
    corpus_words = []
    num_corpus_words = -1
    
    # ------------------
    # Write your implementation here.

    # flatten a list
    # for x in corpus:
    #     for y in x:
    #         corpus_words.append(y)
    corpus_words = list(set([y for x in corpus for y in x]))
    # print(corpus_words)
    
    
    # sort the list
    corpus_words.sort()
    num_corpus_words = len(corpus_words)
    
    # ------------------

    return corpus_words, num_corpus_words

# ---------------------
# Run this sanity check
# Note that this not an exhaustive check for correctness.
# ---------------------

# Define toy corpus
test_corpus = ["START All that glitters isn't gold END".split(" "), "START All's well that ends well END".split(" ")]
test_corpus_words, num_corpus_words = distinct_words(test_corpus)
# print(test_corpus_words, num_corpus_words)

# Correct answers
ans_test_corpus_words = sorted(list(set(["START", "All", "ends", "that", "gold", "All's", "glitters", "isn't", "well", "END"])))
ans_num_corpus_words = len(ans_test_corpus_words)

# Test correct number of words
assert(num_corpus_words == ans_num_corpus_words), "Incorrect number of distinct words. Correct: {}. Yours: {}".format(ans_num_corpus_words, num_corpus_words)

# Test correct words
assert (test_corpus_words == ans_test_corpus_words), "Incorrect corpus_words.\nCorrect: {}\nYours:   {}".format(str(ans_test_corpus_words), str(test_corpus_words))

# Print Success
print ("-" * 80)
print("Passed All Tests!")
print ("-" * 80)

"""Question 1.2: Implement compute_co_occurrence_matrix [code] (3 points)
    Write a method that constructs a co-occurrence matrix for a certain window-size  𝑛
    (with a default of 4), considering words  𝑛  before and  𝑛  after the word in the
    center of the window. Here, we start to use numpy (np) to represent vectors, matrices,
    and tensors. If you're not familiar with NumPy, there's a NumPy tutorial in the second
    half of this cs231n Python NumPy tutorial.
"""

def compute_co_occurrence_matrix(corpus, window_size=4):
    """ Compute co-occurrence matrix for the given corpus and window_size (default of 4).
    
        Note: Each word in a document should be at the center of a window. Words near edges will have a smaller
              number of co-occurring words.
              
              For example, if we take the document "START All that glitters is not gold END" with window size of 4,
              "All" will co-occur with "START", "that", "glitters", "is", and "not".
    
        Params:
            corpus (list of list of strings): corpus of documents
            window_size (int): size of context window
        Return:
            M (numpy matrix of shape (number of corpus words, number of corpus words)): 
                Co-occurence matrix of word counts. 
                The ordering of the words in the rows/columns should be the same as the ordering of the words given by the distinct_words function.
            word2Ind (dict): dictionary that maps word to index (i.e. row/column number) for matrix M.
    """
    words, num_words = distinct_words(corpus)
    M = np.zeros((num_words, num_words))
    word2Ind = dict([(word, index) for index, word in enumerate(words)])
    # ------------------
    # Write your implementation here.
    for sentence in corpus:
        current_index = 0
        sentence_len = len(sentence)
        indices = [word2Ind[i] for i in sentence]
        # print(indices)
        while current_index < sentence_len:
            left  = max(current_index - window_size, 0)
            right = min(current_index + window_size + 1, sentence_len) 
            current_word = sentence[current_index]
            current_word_index = word2Ind[current_word]
            words_around = indices[left:current_index] + indices[current_index+1:right]
            # print(words_around)
            for ind in words_around:
                M[current_word_index, ind] += 1
            
            current_index += 1

    # ------------------
    return M, word2Ind

# ---------------------
# Run this sanity check
# Note that this is not an exhaustive check for correctness.
# ---------------------

# Define toy corpus and get student's co-occurrence matrix
test_corpus = ["START All that glitters isn't gold END".split(" "), "START All's well that ends well END".split(" ")]
M_test, word2Ind_test = compute_co_occurrence_matrix(test_corpus, window_size=1)

# Correct M and word2Ind
M_test_ans = np.array( 
    [[0., 0., 0., 1., 0., 0., 0., 0., 1., 0.,],
     [0., 0., 0., 1., 0., 0., 0., 0., 0., 1.,],
     [0., 0., 0., 0., 0., 0., 1., 0., 0., 1.,],
     [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,],
     [0., 0., 0., 0., 0., 0., 0., 0., 1., 1.,],
     [0., 0., 0., 0., 0., 0., 0., 1., 1., 0.,],
     [0., 0., 1., 0., 0., 0., 0., 1., 0., 0.,],
     [0., 0., 0., 0., 0., 1., 1., 0., 0., 0.,],
     [1., 0., 0., 0., 1., 1., 0., 0., 0., 1.,],
     [0., 1., 1., 0., 1., 0., 0., 0., 1., 0.,]]
)
word2Ind_ans = {'All': 0, "All's": 1, 'END': 2, 'START': 3, 'ends': 4, 'glitters': 5, 'gold': 6, "isn't": 7, 'that': 8, 'well': 9}

# Test correct word2Ind
assert (word2Ind_ans == word2Ind_test), "Your word2Ind is incorrect:\nCorrect: {}\nYours: {}".format(word2Ind_ans, word2Ind_test)

# Test correct M shape
assert (M_test.shape == M_test_ans.shape), "M matrix has incorrect shape.\nCorrect: {}\nYours: {}".format(M_test.shape, M_test_ans.shape)

# Test correct M values
for w1 in word2Ind_ans.keys():
    idx1 = word2Ind_ans[w1]
    for w2 in word2Ind_ans.keys():
        idx2 = word2Ind_ans[w2]
        student = M_test[idx1, idx2]
        correct = M_test_ans[idx1, idx2]
        if student != correct:
            print("Correct M:")
            print(M_test_ans)
            print("Your M: ")
            print(M_test)
            raise AssertionError("Incorrect count at index ({}, {})=({}, {}) in matrix M. Yours has {} but should have {}.".format(idx1, idx2, w1, w2, student, correct))

# Print Success
print ("-" * 80)
print("Passed All Tests!")
print ("-" * 80)

""" Question 1.3: Implement `reduce_to_k_dim` [code] (1 point)

    Construct a method that performs dimensionality reduction on the matrix to produce 
    k-dimensional embeddings. Use SVD to take the top k components and produce a new matrix 
    of k-dimensional embeddings. 

    **Note:** All of numpy, scipy, and scikit-learn (`sklearn`) provide *some* implementation
     of SVD, but only scipy and sklearn provide an implementation of Truncated SVD, and only
      sklearn provides an efficient randomized algorithm for calculating large-scale 
      Truncated SVD. So please use [sklearn.decomposition.TruncatedSVD]
      (https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html).
"""
def reduce_to_k_dim(M, k=2):
    """ Reduce a co-occurence count matrix of dimensionality (num_corpus_words, num_corpus_words)
        to a matrix of dimensionality (num_corpus_words, k) using the following SVD function from Scikit-Learn:
            - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
    
        Params:
            M (numpy matrix of shape (number of corpus words, number of corpus words)): co-occurence matrix of word counts
            k (int): embedding size of each word after dimension reduction
        Return:
            M_reduced (numpy matrix of shape (number of corpus words, k)): matrix of k-dimensioal word embeddings.
                    In terms of the SVD from math class, this actually returns U * S
    """    
    n_iters = 10     # Use this parameter in your call to `TruncatedSVD`
    M_reduced = None
    print("Running Truncated SVD over %i words..." % (M.shape[0]))   

    # ------------------
    # Write your implementation here.
    svd = TruncatedSVD(n_components=k, n_iter=n_iters, random_state=42)
    M_reduced = svd.fit_transform(M)
    # ------------------

    print("Done.")
    return M_reduced

# ---------------------
# Run this sanity check
# Note that this not an exhaustive check for correctness 
# In fact we only check that your M_reduced has the right dimensions.
# ---------------------

# Define toy corpus and run student code
test_corpus = ["START All that glitters isn't gold END".split(" "), "START All's well that ends well END".split(" ")]
M_test, word2Ind_test = compute_co_occurrence_matrix(test_corpus, window_size=1)
M_test_reduced = reduce_to_k_dim(M_test, k=2)

# Test proper dimensions
assert (M_test_reduced.shape[0] == 10), "M_reduced has {} rows; should have {}".format(M_test_reduced.shape[0], 10)
assert (M_test_reduced.shape[1] == 2), "M_reduced has {} columns; should have {}".format(M_test_reduced.shape[1], 2)

# Print Success
print ("-" * 80)
print("Passed All Tests!")
print ("-" * 80)

""" Question 1.4: Implement `plot_embeddings` [code] (1 point)

    Here you will write a function to plot a set of 2D vectors in 2D space. For graphs, 
    we will use Matplotlib (`plt`).

    For this example, you may find it useful to adapt [this code]
    (https://www.pythonmembers.club/2018/05/08/matplotlib-scatter-plot-annotate-set-text-at-label-each-point/). 
    In the future, a good way to make a plot is to look at [the Matplotlib gallery]
    (https://matplotlib.org/gallery/index.html), find a plot that looks somewhat like what you want,
     and adapt the code they give.
"""
def plot_embeddings(M_reduced, word2Ind, words):
    """ Plot in a scatterplot the embeddings of the words specified in the list "words".
        NOTE: do not plot all the words listed in M_reduced / word2Ind.
        Include a label next to each point.
        
        Params:
            M_reduced (numpy matrix of shape (number of unique words in the corpus , k)): matrix of k-dimensioal word embeddings
            word2Ind (dict): dictionary that maps word to indices for matrix M
            words (list of strings): words whose embeddings we want to visualize
    """

    # ------------------
    # Write your implementation here.
    x_coords = M_reduced[:, 0]
    y_coords = M_reduced[:, 1]

    for word in words:
        idx = word2Ind[word]
        embedding = M_reduced[idx]
        x = embedding[0]
        y = embedding[1]
        
        plt.scatter(x, y, marker='x', color='red')
        plt.text(x, y, word, fontsize=9)
    # ------------------
    # plt.show()

# # ---------------------
# # Run this sanity check
# # Note that this not an exhaustive check for correctness.
# # The plot produced should look like the "test solution plot" depicted below. 
# # ---------------------

# print ("-" * 80)
# print ("Outputted Plot:")

# M_reduced_plot_test = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1], [0, 0]])
# word2Ind_plot_test = {'test1': 0, 'test2': 1, 'test3': 2, 'test4': 3, 'test5': 4}
# words = ['test1', 'test2', 'test3', 'test4', 'test5']
# plot_embeddings(M_reduced_plot_test, word2Ind_plot_test, words)

# print ("-" * 80)


# -----------------------------
# Run This Cell to Produce Your Plot
# ------------------------------
reuters_corpus = read_corpus()
M_co_occurrence, word2Ind_co_occurrence = compute_co_occurrence_matrix(reuters_corpus)
M_reduced_co_occurrence = reduce_to_k_dim(M_co_occurrence, k=2)

# Rescale (normalize) the rows to make them each of unit-length
M_lengths = np.linalg.norm(M_reduced_co_occurrence, axis=1)
M_normalized = M_reduced_co_occurrence / M_lengths[:, np.newaxis] # broadcasting

""" Question 1.5: Co-Occurrence Plot Analysis [written] (3 points)

    Now we will put together all the parts you have written! We will compute the co-occurrence
    matrix with fixed window of 4, over the Reuters "crude" corpus. Then we will use TruncatedSVD
    to compute 2-dimensional embeddings of each word. TruncatedSVD returns U\*S, so we normalize
    the returned vectors, so that all the vectors will appear around the unit circle (therefore
    closeness is directional closeness). **Note**: The line of code below that does the 
    normalizing uses the NumPy concept of *broadcasting*. If you don't know about broadcasting, 
    check out [Computation on Arrays: Broadcasting by Jake VanderPlas]
    (https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html).

    Run the below cell to produce the plot. It'll probably take a few seconds to run. What 
    clusters together in 2-dimensional embedding space? What doesn't cluster together that you 
    might think should have?  **Note:** "bpd" stands for "barrels per day" and is a commonly 
    used abbreviation in crude oil topic articles.
"""
words = ['barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'venezuela']
plot_embeddings(M_normalized, word2Ind_co_occurrence, words)


