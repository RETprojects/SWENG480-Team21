from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string


class Preprocessor:
    stemmer = PorterStemmer()
    wnl = WordNetLemmatizer()

    @staticmethod
    def remove_punctuation(input_str):
        """Remove punctuation from a string.

        Keyword arguments:
        input -- a string of any length
        """
        return input_str.translate(str.maketrans('', '', string.punctuation))

    @staticmethod
    def remove_stop_words(input_str):
        """Remove stop words from a string.

        Keyword arguments:
        input -- a string of any length
        """
        return [word.strip() for word in input_str.split(' ') if word and word not in stopwords.words('english')]

    @classmethod
    def do_stem(cls, word_list):
        """Perform Porter stemming on an array of words.

        Keyword arguments:
        input -- an array of strings
        """
        return [cls.stemmer.stem(word.strip()) for word in word_list if word]

    @classmethod
    def lemmatize(cls, word_list):
        """Perform WordNet lemmatization on an array of words.

        Keyword arguments:
        input -- an array of strings
        """
        # Lemmatize for nouns and verbs, and pick the shorter one.
        return [n if len(n) < len(v) else v for word in word_list if word and
                (n := cls.wnl.lemmatize(word.strip()), v := cls.wnl.lemmatize(word.strip(), 'v'))]