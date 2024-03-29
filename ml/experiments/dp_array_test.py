# Remi T, created 11/24/22 12:14 pm CST
# This file contains some code that should loop through an array of design
# problems and perform NLP on them.

# Sources:
#   S. Hussain, J. Keung, M. K. Sohail, A. A. Khan, and M. Ilahi, “Automated
#       Framework for classification and selection of software design
#       patterns,” Applied Soft Computing, vol. 75, pp. 1–20, Feb. 2019.
#   https://www.holisticseo.digital/python-seo/nltk/lemmatize
#   https://www.geeksforgeeks.org/nlp-filtering-insignificant-words/

# imports etc.
import yake
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import collections
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from wordcloud import WordCloud, STOPWORDS

def most_common(lst):
    return max(set(lst), key=lst.count)

# filter out insignificant words like "a" and "and"
def filter_insignificant(chunk,
                         tag_suffixes=['DT', 'CC']):
    good = []

    for word, tag in chunk:
        ok = True

    for suffix in tag_suffixes:
        if tag.endswith(suffix):
            ok = False
            break

        if ok:
            good.append((word, tag))

    return good

# the DPs (lists of design problems)
# For testing purposes, let's assume that all example-oriented language has been filtered out.
gof = [
    "Design a . A design is composed of  (, "
    " and ),  at precise . Each  "
    "form must be modeled by a class that provides a method (): void. A "
    " is a complex  designed by a black-box class component. This "
    "component performs this  in memory, and provides access through a"
    " method get(): int that returns the address of the . It is "
    "probable that the system evolves in order to .",

    "Design a . The  provides  to its "
    "clients with  categories: ,  and . A  is  "
    "during some weeks, and after change category. The  depends on "
    "the category. It is probable that the system evolves in order to take "
    "into account the  category.",

    "Many distinct and unrelated operations need to be performed on node "
    "objects in a heterogeneous aggregate structure. You want to avoid "
    "‘polluting’ the node classes with these operations. And, you do not want"
    " to have to query the type of each node and cast the pointer to the "
    "appropriate type before performing the desired operation"
]
douglass = [
    "One of the key problems with dynamic memory allocation is memory "
    "fragmentation. Memory fragmentation is the random intermixing of free "
    "and allocated memory in the heap. For memory fragmentation to occur, the"
    " two conditions must be met (1) the order of memory allocation is "
    "unrelated to the order in which it is released, and (2) Memory is "
    "allocated in various sizes from the heap.",

    "The problem of deadlock is such a serious one in highly reliable "
    "computing that many systems design in specific mechanisms to detect it "
    "or avoid it. As previously discussed, deadlock occurs when a task is "
    "waiting on a condition that can never, in principle, be satisfied. There"
    " are four conditions that must be true for deadlock to occur, and it is "
    "sufficient to deny the existence of any one of these.",

    "A distributed system using the mailboxes has two IPC primitives, send "
    "and receive. The latter primitive specifics a process to receive from "
    "and blocks if no message from that process is available, even though "
    "message may be waiting from other process. Is deadlock possible, if "
    "there are no shared resources, but process need to communicate frequently."
]
security = [
    "We need to have a way to control access to resources, including "
    "information. The first step is to declare who is authorized to access "
    "resources in specific ways. Otherwise, any active entity (user, process)"
    " could access any resource and we could have confidentiality and "
    "integrity problems. How do we describe who is authorized to access "
    "specific resources in a system?",

    "For convenient administration of authorization rights we need to have "
    "ways to factor out rights. Otherwise, the number of individual rights is"
    " just too large, and granting rights to individual users would require "
    "storing many authorization rules, and it would be hard for "
    "administrators to keep track of these rules. How do we assign rights "
    "based on the functions or tasks of people?",

    "The ability to define an asset’s value is a key component of any risk "
    "assessment. Threats and vulnerabilities that target and expose an asset "
    "are only significant within the context of the asset’s value. Without "
    "this determination, an enterprise is unable to properly assess the risks"
    " posed to its assets. How can an enterprise determine the overall value "
    "of its assets?"
]

# choose an array to be the DP
dp = gof

word_stemmer = PorterStemmer()  # to find the stems of words to account for plural nouns and different tenses of verbs
tokenizer = RegexpTokenizer(r'\w+') # this tokenizer splits up the text into words and filters out punctuation
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()    # lemmatization returns a valid word at all times

for prob in dp:
    # display a word cloud of the words in the text
    wordcloud = WordCloud(stopwords=stop_words, background_color="white", max_words=1000).generate(prob)
    rcParams['figure.figsize'] = 10, 20
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

    # get the individual words of the text, minus extra verb tenses, plurals, and stopwords
    # use lemmatization instead of stemming
    #filtered_words = [most_common([lemmatizer.lemmatize(word.lower(), 'v'),     # verb
    #                               lemmatizer.lemmatize(word.lower(), 'n'),     # noun
    #                               lemmatizer.lemmatize(word.lower(), 'r'),     # adverb
    #                               lemmatizer.lemmatize(word.lower(), 's'),     # satellite adjective
    #                               lemmatizer.lemmatize(word.lower(), 'a')])    # adjective
    #                  for word in tokenizer.tokenize(prob) if word not in stopwords]
    filtered_words = [lemmatizer.lemmatize(word[0].lower(), pos="v") for word in nltk.pos_tag(tokenizer.tokenize(prob))
                      if word[0] not in stop_words]
    counted_words = collections.Counter(filtered_words)
    words = []
    counts = []
    for letter, count in counted_words.most_common(10):
        words.append(letter)
        counts.append(count)
    # display a graph of the 10 most common words in the text
    colors = cm.rainbow(np.linspace(0, 1, 10))
    rcParams['figure.figsize'] = 20, 10
    plt.title('Top words in the headlines vs their count')
    plt.xlabel('Count')
    plt.ylabel('Words')
    plt.barh(words, counts, color=colors)
    plt.show()
