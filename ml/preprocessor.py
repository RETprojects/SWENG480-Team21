from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string


class Preprocessor:
    stemmer = PorterStemmer()


    @staticmethod
    def remove_punctuation(input):
        """Remove punctuation from a string.

        Keyword arguments:
        input -- a string of any length
        """
        return input.translate(str.maketrans('', '', string.punctuation))

    @staticmethod
    def remove_stop_words(input):
        """Remove stop words from a string.

        Keyword arguments:
        input -- a string of any length
        """
        return [word for word in input.split(' ') if word not in stopwords.words('english')]

    @classmethod
    def do_stem(cls, word_list):
        """Perform Porter stemming on an array of words.

        Keyword arguments:
        input -- an array of strings
        """
        return [cls.stemmer.stem(word) for word in word_list]


my_str = "Today I went to the store to buy some groceries. I bought carrots, apples, and bananas."
no_punc = Preprocessor.remove_punctuation(my_str)
no_stop_words = Preprocessor.remove_stop_words(no_punc)
stemmed = Preprocessor.do_stem(no_stop_words)
print(f'No punctuation: {no_punc}\nAnd no stop words: {no_stop_words}')
print(f'Stemmed: {stemmed}')

