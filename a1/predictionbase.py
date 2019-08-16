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

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.vocab.keys())
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin

# -----------------------------------
# Run Cell to Load Word Vectors
# Note: This may take several minutes
# -----------------------------------
wv_from_bin = load_word2vec()

def get_matrix_of_vectors(wv_from_bin, required_words=['barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'venezuela']):
    """ Put the word2vec vectors into a matrix M.
        Param:
            wv_from_bin: KeyedVectors object; the 3 million word2vec vectors loaded from file
        Return:
            M: numpy matrix shape (num words, 300) containing the vectors
            word2Ind: dictionary mapping each word to its row number in M
    """
    import random
    words = list(wv_from_bin.vocab.keys())
    print("Shuffling words ...")
    random.shuffle(words)
    words = words[:10000]
    print("Putting %i words into word2Ind and matrix M..." % len(words))
    word2Ind = {}
    M = []
    curInd = 0
    for w in words:
        try:
            M.append(wv_from_bin.word_vec(w))
            word2Ind[w] = curInd
            curInd += 1
        except KeyError:
            continue
    for w in required_words:
        try:
            M.append(wv_from_bin.word_vec(w))
            word2Ind[w] = curInd
            curInd += 1
        except KeyError:
            continue
    M = np.stack(M)
    print("Done.")
    return M, word2Ind

# -----------------------------------------------------------------
# Run Cell to Reduce 300-Dimensinal Word Embeddings to k Dimensions
# Note: This may take several minutes
# -----------------------------------------------------------------
M, word2Ind = get_matrix_of_vectors(wv_from_bin)
M_reduced = reduce_to_k_dim(M, k=2)

""" Question 2.1: Word2Vec Plot Analysis [written] (4 points)
    Run the cell below to plot the 2D word2vec embeddings for ['barrels', 'bpd', 'ecuador', 
    'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'venezuela'].

    What clusters together in 2-dimensional embedding space? What doesn't cluster together 
    that you might think should have? How is the plot different from the one generated earlier 
    from the co-occurrence matrix?
"""
# words = ['barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'venezuela']
# plot_embeddings(M_reduced, word2Ind, words)

""" Question 2.2: Polysemous Words (2 points) [code + written] 
    Find a [polysemous](https://en.wikipedia.org/wiki/Polysemy) word 
    (for example, "leaves" or "scoop") such that the top-10 most similar words 
    (according to cosine similarity) contains related words from *both* meanings. 
    For example, "leaves" has both "vanishes" and "stalks" in the top 10, and "scoop" has both 
    "handed_waffle_cone" and "lowdown". You will probably need to try several polysemous words 
    before you find one. Please state the polysemous word you discover and the multiple meanings 
    at occur in the top 10. Why do you think many of the polysemous words you tried didn't work?

    **Note**: You should use the `wv_from_bin.most_similar(word)` function to get the top 10 
    similar words. This function ranks all other words in the vocabulary with respect to their 
    cosine similarity to the given word. For further assistance please check the __
    [GenSim documentation](https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.FastTextKeyedVectors.most_similar)__.
"""

# ------------------
# Write your polysemous word exploration code here.

wv_from_bin.most_similar("leaves")

# ------------------

""" Question 2.3: Synonyms & Antonyms (2 points) [code + written] 

    When considering Cosine Similarity, it's often more convenient to think of Cosine Distance, 
    which is simply 1 - Cosine Similarity.

    Find three words (w1,w2,w3) where w1 and w2 are synonyms and w1 and w3 are antonyms, but 
    Cosine Distance(w1,w3) < Cosine Distance(w1,w2). For example, w1="happy" is closer to w3="sad"
    than to w2="cheerful". 

    Once you have found your example, please give a possible explanation for why this counter-
    intuitive result may have happened.

    You should use the the `wv_from_bin.distance(w1, w2)` function here in order to compute the 
    cosine distance between two words. Please see the __[GenSim documentation]
    (https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.FastTextKeyedVectors.distance)__ 
    for further assistance.
"""

# ------------------
# Write your synonym & antonym exploration code here.

w1 = "happy"
w2 = "cheerful"
w3 = "sad"
w1_w2_dist = wv_from_bin.distance(w1, w2)
w1_w3_dist = wv_from_bin.distance(w1, w3)

print("Synonyms {}, {} have cosine distance: {}".format(w1, w2, w1_w2_dist))
print("Antonyms {}, {} have cosine distance: {}".format(w1, w3, w1_w3_dist))

# ------------------


""" Solving Analogies with Word Vectors
    Word2Vec vectors have been shown to *sometimes* exhibit the ability to solve analogies. 

    As an example, for the analogy "man : king :: woman : x", what is x?

    In the cell below, we show you how to use word vectors to find x. The `most_similar` 
    function finds words that are most similar to the words in the `positive` list and most 
    dissimilar from the words in the `negative` list. The answer to the analogy will be the 
    word ranked most similar (largest numerical value).

    **Note:** Further Documentation on the `most_similar` function can be found within the __
    [GenSim documentation](https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.FastTextKeyedVectors.most_similar)__.
"""

# Run this cell to answer the analogy -- man : king :: woman : x
pprint.pprint(wv_from_bin.most_similar(positive=['woman', 'king'], negative=['man']))

"""Question 2.4: Finding Analogies [code + written] (2 Points)
    Find an example of analogy that holds according to these vectors (i.e. the intended 
    word is ranked top). In your solution please state the full analogy in the form x:y :: a:b. 
    If you believe the analogy is complicated, explain why the analogy holds in one or two 
    sentences.

    Note: You may have to try many analogies to find one that works!
"""

# ------------------
# Write your analogy exploration code here.

pprint.pprint(wv_from_bin.most_similar(positive=['winter', 'hot'], negative=['summer']))

# ------------------

# Question 2.5: Incorrect Analogy [code + written] (1 point)
pprint.pprint(wv_from_bin.most_similar(positive=['computer', 'brain'], negative=['human']))

""" Question 2.6: Guided Analysis of Bias in Word Vectors [written] (1 point)

    It's important to be cognizant of the biases (gender, race, sexual orientation etc.) 
    implicit to our word embeddings.

    Run the cell below, to examine (a) which terms are most similar to "woman" and "boss" and 
    most dissimilar to "man", and (b) which terms are most similar to "man" and "boss" and most 
    dissimilar to "woman". What do you find in the top 10?
"""
# Run this cell
# Here `positive` indicates the list of words to be similar to and `negative` indicates the list of words to be
# most dissimilar from.
pprint.pprint(wv_from_bin.most_similar(positive=['woman', 'boss'], negative=['man']))
print()
pprint.pprint(wv_from_bin.most_similar(positive=['man', 'boss'], negative=['woman']))

# Question 2.7: Independent Analysis of Bias in Word Vectors
# Use the most_similar function to find another case where some bias is exhibited by the vectors. 
# Please briefly explain the example of bias that you discover.

# ------------------
# Write your bias exploration code here.
np.warnings.filterwarnings('ignore')
print("black:waving :: white: ?")
pprint.pprint(wv_from_bin.most_similar(positive=['white', 'waving'], negative=['black']))
print("white:waving :: black: ?")
pprint.pprint(wv_from_bin.most_similar(positive=['black', 'waving'], negative=['white']))

# ------------------

