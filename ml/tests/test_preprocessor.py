import unittest

from ml.preprocessor import Preprocessor


# (venv) jonathangarelick@Jonathans-MacBook-Pro SWENG480-Team21 % python -m unittest -v ml/tests/test_preprocessor.py

class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        self.str_simple = "Today I went to the store to buy some groceries. I bought carrots, apples, and bananas."
        self.str_complex = """Natural language processing (NLP) is a subfield of linguistics, computer science,
            and artificial intelligence concerned with the interactions between computers and human language,
            in particular how to program computers to process and analyze large amounts of natural language data.
            The goal is a computer capable of "understanding" the contents of documents, including the contextual
            nuances of the language within them. The technology can then accurately extract information and insights
            contained in the documents as well as categorize and organize the documents themselves.
            """

    def test_remove_punctuation(self):
        self.assertEqual(
            Preprocessor.remove_punctuation(self.str_simple),
            'Today I went to the store to buy some groceries I bought carrots apples and bananas'
        )
        self.assertEqual(
            Preprocessor.remove_punctuation(self.str_complex),
            """Natural language processing NLP is a subfield of linguistics computer science
            and artificial intelligence concerned with the interactions between computers and human language
            in particular how to program computers to process and analyze large amounts of natural language data
            The goal is a computer capable of understanding the contents of documents including the contextual
            nuances of the language within them The technology can then accurately extract information and insights
            contained in the documents as well as categorize and organize the documents themselves
            """
        )

    def test_remove_stop_words(self):
        self.assertEqual(
            Preprocessor.remove_stop_words(self.str_simple),
            ['Today', 'I', 'went', 'store', 'buy', 'groceries.', 'I', 'bought', 'carrots,', 'apples,', 'bananas.']
        )
        self.assertEqual(
            Preprocessor.remove_stop_words(self.str_complex),
            ['Natural', 'language', 'processing', '(NLP)', 'subfield', 'linguistics,', 'computer', 'science,',
             'artificial', 'intelligence', 'concerned', 'interactions', 'computers', 'human', 'language,', 'particular',
             'program', 'computers', 'process', 'analyze', 'large', 'amounts', 'natural', 'language', 'data.', 'The',
             'goal', 'computer', 'capable', '"understanding"', 'contents', 'documents,', 'including', 'contextual',
             'nuances', 'language', 'within', 'them.', 'The', 'technology', 'accurately', 'extract', 'information',
             'insights', 'contained', 'documents', 'well', 'categorize', 'organize', 'documents', 'themselves.']
        )

    def test_do_stem(self):
        self.assertEqual(
            Preprocessor.do_stem(self.str_simple.split(' ')),
            ['today', 'i', 'went', 'to', 'the', 'store', 'to', 'buy', 'some', 'groceries.', 'i', 'bought', 'carrots,',
             'apples,', 'and', 'bananas.']
        )
        self.assertEqual(
            Preprocessor.do_stem(self.str_complex.split(' ')),
            ['natur', 'languag', 'process', '(nlp)', 'is', 'a', 'subfield', 'of', 'linguistics,', 'comput', 'science,',
             'and', 'artifici', 'intellig', 'concern', 'with', 'the', 'interact', 'between', 'comput', 'and', 'human',
             'language,', 'in', 'particular', 'how', 'to', 'program', 'comput', 'to', 'process', 'and', 'analyz',
             'larg', 'amount', 'of', 'natur', 'languag', 'data.', 'the', 'goal', 'is', 'a', 'comput', 'capabl', 'of',
             '"understanding"', 'the', 'content', 'of', 'documents,', 'includ', 'the', 'contextu', 'nuanc', 'of', 'the',
             'languag', 'within', 'them.', 'the', 'technolog', 'can', 'then', 'accur', 'extract', 'inform', 'and',
             'insight', 'contain', 'in', 'the', 'document', 'as', 'well', 'as', 'categor', 'and', 'organ', 'the',
             'document', 'themselves.']

        )

    # Lemmatization breaks if punctuation is not removed.
    def test_lemmatize(self):
        self.assertEqual(
            Preprocessor.lemmatize(Preprocessor.remove_punctuation(self.str_simple).split(' ')),
            ['Today', 'I', 'went', 'to', 'the', 'store', 'to', 'buy', 'some', 'grocery', 'I', 'bought', 'carrot',
             'apple', 'and', 'banana']
        )
        self.assertEqual(
            Preprocessor.lemmatize(Preprocessor.remove_punctuation(self.str_complex).split(' ')),
            ['Natural', 'language', 'processing', 'NLP', 'is', 'a', 'subfield', 'of', 'linguistics', 'computer',
             'science', 'and', 'artificial', 'intelligence', 'concerned', 'with', 'the', 'interaction', 'between',
             'computer', 'and', 'human', 'language', 'in', 'particular', 'how', 'to', 'program', 'computer', 'to',
             'process', 'and', 'analyze', 'large', 'amount', 'of', 'natural', 'language', 'data', 'The', 'goal', 'is',
             'a', 'computer', 'capable', 'of', 'understanding', 'the', 'content', 'of', 'document', 'including', 'the',
             'contextual', 'nuance', 'of', 'the', 'language', 'within', 'them', 'The', 'technology', 'can', 'then',
             'accurately', 'extract', 'information', 'and', 'insight', 'contained', 'in', 'the', 'document', 'a',
             'well', 'a', 'categorize', 'and', 'organize', 'the', 'document', 'themselves']
        )


if __name__ == '__main__':
    unittest.main()
