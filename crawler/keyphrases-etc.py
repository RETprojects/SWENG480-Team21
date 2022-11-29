# Remi T, created 11/19/22 3:02 pm CST
# This file contains some code that can be used for machine learning.

#Sources: https://www.tutorialspoint.com/natural_language_toolkit/natural_language_toolkit_quick_guide.htm
#   https://www.analyticsvidhya.com/blog/2022/01/four-of-the-easiest-and-most-effective-methods-of-keyword-extraction-from-a-single-text-using-python/
#   https://www.analyticsvidhya.com/blog/2022/03/keyword-extraction-methods-from-documents-in-nlp/
#   https://stackoverflow.com/a/15555162
#   https://stackoverflow.com/a/45384376
#   https://www.holisticseo.digital/python-seo/nltk/lemmatize
#   https://www.nltk.org/book_1ed/ch05.html
#   https://www.geeksforgeeks.org/python-lemmatization-with-nltk/

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

"""
text = "Imagine that you’re creating a logistics management application. " \
       "The first version of your app can only handle transportation by " \
       "trucks, so the bulk of your code lives inside the Truck class. " \
       "After a while, your app becomes pretty popular. Each day you " \
       "receive dozens of requests from sea transportation companies to " \
       "incorporate sea logistics into the app. Adding a new " \
       "transportation class to the program causes an issue. Adding a new " \
       "class to the program isn’t that simple if the rest of the code is " \
       "already coupled to existing classes. Great news, right? But how " \
       "about the code? At present, most of your code is coupled to the " \
       "Truck class. Adding Ships into the app would require making " \
       "changes to the entire codebase. Moreover, if later you decide to " \
       "add another type of transportation to the app, you will probably " \
       "need to make all of these changes again. As a result, you will end " \
       "up with pretty nasty code, riddled with conditionals that switch " \
       "the app’s behavior depending on the class of transportation objects."
"""
# the same text without as many example-oriented words
text = "Imagine that you’re creating a  application. " \
       "The first version of your app can only handle  by " \
       ", so the bulk of your code lives inside the  class. " \
       "After a while, your app becomes  popular. Each day you " \
       "receive dozens of requests  to " \
       "incorporate  into the app. Adding a new " \
       " class to the program causes an issue. Adding a new " \
       "class to the program isn’t that simple if the rest of the code is " \
       "already coupled to existing classes. Great news, right? But how " \
       "about the code? At present, most of your code is coupled to the " \
       " class. Adding  into the app would require making " \
       "changes to the entire codebase. Moreover, if later you decide to " \
       "add another type of  to the app, you will probably " \
       "need to make all of these changes again. As a result, you will end " \
       "up with  nasty code, riddled with conditionals that switch " \
       "the app’s behavior depending on the class of  objects."

def most_common(lst):
    return max(set(lst), key=lst.count)

# return the most basic form a word based on part of speech
def basicForm(word):
    lemmatizer = WordNetLemmatizer()

    if word[0][1] in ['V','VD','VG','VN']:
        return lemmatizer.lemmatize(word, "v")
    else:
        return lemmatizer.lemmatize(word)

# using NLTK
english_stops = set(stopwords.words('english'))
words = word_tokenize(text)
words_list = [word for word in words if word not in english_stops]
print(words_list)
important_words = ' '.join(words_list)
print(important_words)

# print the top 10 keyphrases in the text
# using YAKE!
kw_extractor = yake.KeywordExtractor(top=10, stopwords=None)
keyphrases = kw_extractor.extract_keywords(important_words)
for kw, v in keyphrases:
    print("Keyphrase: ",kw, ": score", v)

# NLTK information extraction
sentences = nltk.sent_tokenize(text)
tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
for sent in tagged_sentences:
    print(nltk.ne_chunk(sent))

word_stemmer = PorterStemmer()  # to find the stems of words to account for plural nouns and different tenses of verbs
tokenizer = RegexpTokenizer(r'\w+') # this tokenizer splits up the text into words and filters out punctuation
#stopwords = STOPWORDS
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()    # lemmatization returns a valid word at all times

# display a word cloud of the words in the text
wordcloud = WordCloud(stopwords=stop_words, background_color="white", max_words=1000).generate(text)
rcParams['figure.figsize'] = 10, 20
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

# get the individual words of the text, minus extra verb tenses, plurals, and stopwords
# use lemmatization instead of stemming
#filtered_words = [most_common([lemmatizer.lemmatize(word.lower(),'v'),lemmatizer.lemmatize(word.lower(),'n'),lemmatizer.lemmatize(word.lower(),'n')]) for word in tokenizer.tokenize(text) if word not in stopwords]
#filtered_words = [most_common([lemmatizer.lemmatize(word.lower(), 'a'),     # adjective
#                                lemmatizer.lemmatize(word.lower(), 's'),    # satellite adjective
#                                lemmatizer.lemmatize(word.lower(), 'r'),    # adverb
#                                lemmatizer.lemmatize(word.lower(), 'n'),    # noun
#                                lemmatizer.lemmatize(word.lower(), 'v')])   # verb
#                    for word in tokenizer.tokenize(text) if word not in stopwords]
#filtered_words = [basicForm(word.lower()) for word in tokenizer.tokenize(text) if word not in stopwords]
filtered_words = [lemmatizer.lemmatize(word[0].lower(), pos="v") for word in nltk.pos_tag(tokenizer.tokenize(text)) if word[0] not in stop_words]
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
# now print each word in counted_words for validation
print(counted_words)
# another test
for word in tokenizer.tokenize(text):
    checkList=[lemmatizer.lemmatize(word,'v'),lemmatizer.lemmatize(word,'n'),lemmatizer.lemmatize(word,'n')]
    print(checkList)
