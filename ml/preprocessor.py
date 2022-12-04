from nltk.corpus import stopwords


class Preprocessor:
    def __init__(self):
        word_list = ['Today', 'I', 'went', 'to', 'the', 'store', 'to', 'buy', 'some', 'groceries.', 'I', 'bought',
                     'carrots,', 'apples,', 'and', 'bananas.']
        filtered_words = [word for word in word_list if word not in stopwords.words('english')]
        print(filtered_words)


Preprocessor()
