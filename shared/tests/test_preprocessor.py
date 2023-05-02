import unittest

from shared.preprocessor import Preprocessor


# python -m unittest -v shared/tests/test_preprocessor.py

class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        self.str_simple = "Today I went to the store to buy some groceries. I bought carrots, apples, and bananas."
        self.str_simple_nopunc = "Today I went to the store to buy some groceries I bought carrots apples and bananas"
        self.str_complex = """Natural language processing (NLP) is a subfield of linguistics, computer science,
            and artificial intelligence concerned with the interactions between computers and human language,
            in particular how to program computers to process and analyze large amounts of natural language data.
            The goal is a computer capable of "understanding" the contents of documents, including the contextual
            nuances of the language within them. The technology can then accurately extract information and insights
            contained in the documents as well as categorize and organize the documents themselves.
            """
        self.str_complex_nopunc = """Natural language processing NLP is a subfield of linguistics computer science
            and artificial intelligence concerned with the interactions between computers and human language
            in particular how to program computers to process and analyze large amounts of natural language data
            The goal is a computer capable of understanding the contents of documents including the contextual
            nuances of the language within them The technology can then accurately extract information and insights
            contained in the documents as well as categorize and organize the documents themselves
            """

    def test_remove_punctuation(self):
        self.assertEqual(
            Preprocessor.remove_punctuation(self.str_simple),
            self.str_simple_nopunc
        )
        self.assertEqual(
            Preprocessor.remove_punctuation(self.str_complex),
            self.str_complex_nopunc
        )

    def test_remove_stop_words(self):
        self.assertEqual(
            Preprocessor.remove_stop_words(self.str_simple_nopunc),
            ['today', 'went', 'store', 'buy', 'groceries', 'bought', 'carrots', 'apples', 'bananas']
        )
        self.assertEqual(
            Preprocessor.remove_stop_words(self.str_complex_nopunc),
            ['natural', 'language', 'processing', 'nlp', 'subfield', 'linguistics', 'computer', 'science',
             'artificial', 'intelligence', 'concerned', 'interactions', 'computers', 'human', 'language',
             'particular', 'program', 'computers', 'process', 'analyze', 'large', 'amounts', 'natural',
             'language', 'data', 'goal', 'capable', 'understanding', 'contents', 'documents',
             'including', 'contextual', 'nuances', 'language', 'within', 'technology', 'accurately', 'extract',
             'information', 'insights', 'contained', 'documents', 'well', 'categorize', 'organize', 'documents']
        )

    def test_do_stem(self):
        self.assertEqual(
            Preprocessor.do_stem(self.str_simple_nopunc.split()),
            ['today', 'i', 'went', 'to', 'the', 'store', 'to', 'buy', 'some', 'groceri', 'i', 'bought', 'carrot',
             'appl', 'and', 'banana']
        )
        self.assertEqual(
            Preprocessor.do_stem(self.str_complex_nopunc.split()),
            ['natur', 'languag', 'process', 'nlp', 'is', 'a', 'subfield', 'of', 'linguist', 'comput', 'scienc',
             'and', 'artifici', 'intellig', 'concern', 'with', 'the', 'interact', 'between', 'comput', 'and',
             'human', 'languag', 'in', 'particular', 'how', 'to', 'program', 'comput', 'to', 'process', 'and',
             'analyz', 'larg', 'amount', 'of', 'natur', 'languag', 'data', 'the', 'goal', 'is', 'a', 'comput',
             'capabl', 'of', 'understand', 'the', 'content', 'of', 'document', 'includ', 'the', 'contextu', 'nuanc',
             'of', 'the', 'languag', 'within', 'them', 'the', 'technolog', 'can', 'then', 'accur', 'extract', 'inform',
             'and', 'insight', 'contain', 'in', 'the', 'document', 'as', 'well', 'as', 'categor', 'and', 'organ', 'the',
             'document', 'themselv']

        )

    def test_lemmatize(self):
        self.assertEqual(
            Preprocessor.lemmatize(self.str_simple_nopunc.split(' ')),
            ['Today', 'I', 'go', 'to', 'the', 'store', 'to', 'buy', 'some', 'grocery', 'I', 'buy', 'carrot', 'apple',
             'and', 'banana']
        )
        self.assertEqual(
            Preprocessor.lemmatize(self.str_complex_nopunc.split(' ')),
            ['Natural', 'language', 'process', 'NLP', 'be', 'a', 'subfield', 'of', 'linguistics', 'computer', 'science',
             'and', 'artificial', 'intelligence', 'concern', 'with', 'the', 'interaction', 'between', 'computer', 'and',
             'human', 'language', 'in', 'particular', 'how', 'to', 'program', 'computer', 'to', 'process', 'and',
             'analyze', 'large', 'amount', 'of', 'natural', 'language', 'data', 'The', 'goal', 'be', 'a', 'computer',
             'capable', 'of', 'understand', 'the', 'content', 'of', 'document', 'include', 'the', 'contextual',
             'nuance', 'of', 'the', 'language', 'within', 'them', 'The', 'technology', 'can', 'then', 'accurately',
             'extract', 'information', 'and', 'insight', 'contain', 'in', 'the', 'document', 'a', 'well', 'a',
             'categorize', 'and', 'organize', 'the', 'document', 'themselves']

        )

    def test_integration(self):
        self.assertEqual(
            Preprocessor.do_stem(
                Preprocessor.remove_stop_words(
                    Preprocessor.remove_punctuation(self.str_simple)
                )
            ),
            ['today', 'went', 'store', 'buy', 'groceri', 'bought', 'carrot', 'appl', 'banana']
        )
        self.assertEqual(
            Preprocessor.do_stem(
                Preprocessor.remove_stop_words(
                    Preprocessor.remove_punctuation(self.str_complex)
                )
            ),
            ['natur', 'languag', 'process', 'nlp', 'subfield', 'linguist', 'comput', 'scienc',
             'artifici', 'intellig', 'concern', 'interact', 'comput', 'human', 'languag', 'particular',
             'program', 'comput', 'process', 'analyz', 'larg', 'amount', 'natur', 'languag', 'data', 'goal',
             'comput', 'capabl', 'understand', 'content', 'document', 'includ', 'contextu', 'nuanc', 'languag',
             'within', 'technolog', 'accur', 'extract', 'inform', 'insight', 'contain', 'document', 'well',
             'categor', 'organ', 'document']
        )


if __name__ == '__main__':
    unittest.main()
