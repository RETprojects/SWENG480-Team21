# Remi T, created 11/19/22 3:02 pm CST
# This file contains some code that can be used for machine learning.

import yake
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

text = "Imagine that you’re creating a logistics management application. The first version of your app can only handle transportation by trucks, so the bulk of your code lives inside the Truck class. After a while, your app becomes pretty popular. Each day you receive dozens of requests from sea transportation companies to incorporate sea logistics into the app. Adding a new transportation class to the program causes an issue Adding a new class to the program isn’t that simple if the rest of the code is already coupled to existing classes. Great news, right? But how about the code? At present, most of your code is coupled to the Truck class. Adding Ships into the app would require making changes to the entire codebase. Moreover, if later you decide to add another type of transportation to the app, you will probably need to make all of these changes again. As a result, you will end up with pretty nasty code, riddled with conditionals that switch the app’s behavior depending on the class of transportation objects."

# using NLTK
english_stops = set(stopwords.words('english'))
words = word_tokenize(text)
keywords_list = [word for word in words if word not in english_stops]
print(keywords_list)
keywords = ' '.join(keywords_list)
print(keywords)

# print the top 10 keywords in the text
# using YAKE!
kw_extractor = yake.KeywordExtractor(top=10, stopwords=None)
keywords = kw_extractor.extract_keywords(keywords)
for kw, v in keywords:
  print("Keyphrase: ",kw, ": score", v)

# NLTK information extraction
sentences = nltk.sent_tokenize(text)
tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
for sent in tagged_sentences:
  print(nltk.ne_chunk(sent))
