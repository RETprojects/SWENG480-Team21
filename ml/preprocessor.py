from nltk.corpus import stopwords
import string


class Preprocessor:
    @staticmethod
    def remove_punctuation(input):
        return input.translate(str.maketrans('', '', string.punctuation))

    @staticmethod
    def remove_stop_words(input):
        return [word for word in input.split(' ') if word not in stopwords.words('english')]


my_str = "Today I went to the store to buy some groceries. I bought carrots, apples, and bananas."
no_punc = Preprocessor.remove_punctuation(my_str)
no_stop_words = Preprocessor.remove_stop_words(no_punc)
print(f'No punctuation: {no_punc}\nAnd no stop words: {no_stop_words}')

