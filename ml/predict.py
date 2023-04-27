# Command line usage: `python predict.py "Insert design problem here."`

import os
import sys

import nltk
import numpy as np
import pandas as pd
from fcmeans import FCM
from nltk import PorterStemmer, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from sklearn import cluster
from sklearn.cluster import AgglomerativeClustering, BisectingKMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn_extra.cluster import KMedoids
from unidecode import unidecode
import re

try:
    nltk.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# global variables
algorithms = [
    "kmeans",
    "fuzzy_cmeans",
    "agglomerative",
    "bi_kmeans_inertia",
    "bi_kmeans_lg_cluster",
    "pam_euclidean",
    "pam_manhattan",
]

algorithms_pretty = [
    "K-means",
    "Fuzzy c-means",
    "Agglomerative",
    "Bisecting k-means (biggest inertia strategy)",
    "Bisecting k-means (largest cluster strategy)",
    "K-medoids/PAM (Euclidean distance)",
    "K-medoids/PAM (Manhattan distance)",
]

stemmer = PorterStemmer()

# This is for the Django web app. This can definitely be improved.
output = []


def do_output(text: str = "") -> None:
    output.append(text)
    print(text)


def preprocess(series: pd.Series) -> pd.Series:
    # Lowercase
    series = series.str.lower()
    # Remove all non-alphabetical characters
    series = series.str.replace("[^A-Za-z]", repl=" ", regex=True)
    # Stem, remove stop words, and remove all parts of speech (POS) except verbs and adjectives
    series = series.map(
        lambda x: " ".join(
            [
                stemmer.stem(word)
                for word, pos in pos_tag(x.split(), tagset="universal")
                if word not in stop_words and (pos == "VERB" or pos == "ADJ")
            ]
        )
    )
    # Remove leading and trailing space
    series = series.str.strip()

    return series


def do_cluster(df_weighted: pd.DataFrame, algo: str) -> pd.DataFrame:
    # This is the DataFrame we will return that contains all the labels.
    df = pd.DataFrame()

    if algo == "agglomerative":
        # Agglomerative (hierarchical)
        agg = AgglomerativeClustering(n_clusters=3)
        df["agglomerative"] = agg.fit_predict(df_weighted)
    elif algo == "fuzzy_cmeans":
        # Fuzzy c-means
        final_df_np = df_weighted.to_numpy()
        fcm = FCM(n_clusters=3, m=1.67, max_iter=270)
        fcm.fit(final_df_np)
        df["fuzzy_cmeans"] = fcm.predict(final_df_np)
    elif algo == "kmeans":
        # K-means
        km = cluster.KMeans(n_clusters=3, n_init="auto")
        df["kmeans"] = km.fit_predict(df_weighted)
    elif algo == "bi_kmeans_inertia":
        # Bisecting k-means (inertia)
        bisect_inertia = BisectingKMeans(n_clusters=3)
        df["bi_kmeans_inertia"] = bisect_inertia.fit_predict(df_weighted)
    elif algo == "bi_kmeans_lg_cluster":
        # Bisecting k-means (largest cluster)
        bisect_lg_cluster = BisectingKMeans(
            n_clusters=3, bisecting_strategy="largest_cluster"
        )
        df["bi_kmeans_lg_cluster"] = bisect_lg_cluster.fit_predict(df_weighted)
    elif algo == "pam_euclidean":
        # K-medoids (Euclidean distance)
        kmed_euclidean = KMedoids(n_clusters=3)
        df["pam_euclidean"] = kmed_euclidean.fit_predict(df_weighted)
    elif algo == "pam_manhattan":
        # K-medoids (Manhattan distance)
        kmed_manhattan = KMedoids(n_clusters=3, metric="manhattan")
        df["pam_manhattan"] = kmed_manhattan.fit_predict(df_weighted)
    else:
        print("ERROR: Invalid algorithm!")

    return df


# Better for this to be an enum, but the syntax is a bit tricky.
weighting_methods = {"Binary", "Count", "Tfidf"}

# removes a list of words (ie. stopwords) from a tokenized list.
def removeWords(listOfTokens, listOfWords):
    return [token for token in listOfTokens if token not in listOfWords]


# applies stemming to a list of tokenized words
def applyStemming(listOfTokens, stemmer):
    return [stemmer.stem(token) for token in listOfTokens]


# applied lemmatization to a list of tokenized words
def applyLemmatization(listOfTokens, lemmatizer):
    return [lemmatizer.lemmatize(token) for token in listOfTokens]


# removes any words composed of less than 2 or more than 21 letters
def twoLetters(listOfTokens):
    twoLetterWord = []
    for token in listOfTokens:
        if len(token) <= 2 or len(token) >= 21:
            twoLetterWord.append(token)
    return twoLetterWord


def processCorpus(corpus, language, stemmer):
    stopwords = nltk.corpus.stopwords.words(language)
    param_stemmer = stemmer

    for document in corpus:
        index = corpus.index(document)
        corpus[index] = str(corpus[index]).replace(
            "\ufffd", "8"
        )  # Replaces the ASCII '�' symbol with '8'
        corpus[index] = corpus[index].replace(",", "")  # Removes commas
        corpus[index] = corpus[index].rstrip("\n")  # Removes line breaks
        corpus[index] = corpus[index].casefold()  # Makes all letters lowercase

        corpus[index] = re.sub(
            "\W_", " ", corpus[index]
        )  # removes specials characters and leaves only words
        corpus[index] = re.sub(
            "\S*\d\S*", " ", corpus[index]
        )  # removes numbers and words concatenated with numbers IE h4ck3r. Removes road names such as BR-381.
        corpus[index] = re.sub(
            "\S*@\S*\s?", " ", corpus[index]
        )  # removes emails and mentions (words with @)
        corpus[index] = re.sub(r"http\S+", "", corpus[index])  # removes URLs with http
        corpus[index] = re.sub(r"www\S+", "", corpus[index])  # removes URLs with www

        listOfTokens = word_tokenize(corpus[index])
        twoLetterWord = twoLetters(listOfTokens)

        listOfTokens = removeWords(listOfTokens, stopwords)
        listOfTokens = removeWords(listOfTokens, twoLetterWord)

        listOfTokens = applyStemming(listOfTokens, param_stemmer)

        corpus[index] = " ".join(listOfTokens)
        corpus[index] = unidecode(corpus[index])

    return corpus


# Output: DataFrame with dense values
def do_weighting(method: str, series: pd.Series) -> pd.DataFrame:
    if method == "Binary":
        vectorizer = CountVectorizer(binary=True)
    elif method == "Count":
        vectorizer = CountVectorizer()
    elif method == "Tfidf":
        vectorizer = TfidfVectorizer(sublinear_tf=True)
    else:
        print("Error. Did not pass valid weighting method")
        return

    matrix = vectorizer.fit_transform(series)
    return pd.DataFrame.sparse.from_spmatrix(
        matrix, columns=vectorizer.get_feature_names_out()
    ).sparse.to_dense()


def file_to_df(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".xls") or path.endswith(".xlsx"):
        df = pd.read_excel(path)
    return df


# https://stackoverflow.com/a/20007730
def ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = ["th", "st", "nd", "rd", "th"][min(n % 10, 4)]
    return str(n) + suffix


# Source: https://danielcaraway.github.io/html/sklearn_cosine_similarity.html
def cosine_sim(df, df_col, class_no, pos_to_last, predicted_labels):
    unigram_count = CountVectorizer(encoding="latin-1", binary=False)
    unigram_count_stop_remove = CountVectorizer(
        encoding="latin-1", binary=False, stop_words="english"
    )

    # get the list of candidate patterns
    txts = df_col.loc[predicted_labels == class_no]  # where label == class_no
    vecs = unigram_count.fit_transform(txts)

    cos_sim = cosine_similarity(vecs[-pos_to_last], vecs)

    return cos_sim, txts


# Written by Akash
class PredictedPattern:
    allPatternsPredicted = []
    # "Stores name and reccurence "

    def __init__(self, name, timesPredicted, cosineSimPercent):
        self.name = name
        self.timesPredicted = timesPredicted
        self.cosineSimPercent = cosineSimPercent
        self.clusterPercent = 0
        self.totalPercent = 0

    def assignPercentages(self, clusterPercent, totalPercent):
        self.clusterPercent = clusterPercent
        self.totalPercent = totalPercent


# Calculates three different percent match metrics between a design problem and a design pattern
# Written by Akash
def CalculatePercent(predictedPattern, clusterWeight, cosSimWeight, times_clustered):
    totalPercent = 0
    clusterPercent = 0
    cosSimPercent = 0

    clusterPercent = int((predictedPattern.timesPredicted / times_clustered) * 100)
    cosSimPercent = predictedPattern.cosineSimPercent
    totalPercent = int(
        (((clusterPercent * clusterWeight) + (cosSimPercent * cosSimWeight)) / 2)
        + ((2 - clusterWeight - cosSimWeight) * 70)
    )

    return clusterPercent, cosSimPercent, totalPercent


def main(design_problem: str = ""):
    num_of_times_clustered = 10
    clusterWeight = 0.9
    cossimWeight = 0.8

    # Reset output
    output.clear()

    # Handle command line execution
    if not design_problem:
        design_problem = sys.argv[1]

    df = file_to_df(
        os.path.join(os.path.dirname(__file__), f"data/GOF Patterns (2.0).csv")
    )

    if df.empty:
        print("Error reading file.")
        return

    # Final example demonstrates how to append a Series as a row
    # https://pandas.pydata.org/docs/reference/api/pandas.concat.html
    new_row = pd.Series(
        {
            "name": "design_problem",
            "overview": design_problem,
        }
    )
    df = pd.concat([df, new_row.to_frame().T], ignore_index=True)

    # # Preprocess
    # cleaned_text = preprocess(df["overview"])
    # # Create a dense tfidf matrix
    # tfidf_matrix = do_weighting("Tfidf", cleaned_text)

    # pre process
    corpus = df["overview"].tolist()
    corpus = processCorpus(corpus, language="english", stemmer=stemmer)

    # vectorize data
    vectorizer = TfidfVectorizer(sublinear_tf=True)
    X = vectorizer.fit_transform(corpus)
    tfidf_matrix = pd.DataFrame(
        data=X.toarray(), columns=vectorizer.get_feature_names_out()
    )
    # Perform clustering
    for a, algo in enumerate(algorithms):
        predictedPatterns = []
        PredictedPattern.allPatternsPredicted.clear()

        for j in range(num_of_times_clustered):
            df_labels = do_cluster(tfidf_matrix, algo)
            df[algo] = df_labels

            cos_sim, txts = cosine_sim(
                df, df["overview"], df[algo].iloc[df.index[-1]], 1, df[algo]
            )

            # get problem row
            n = df.index[-1]
            problemRow = df.iloc[[n]]

            sim_sorted_doc_idx = cos_sim.argsort()

            # loop patterns of the same cluster as the design problem
            for i in range(len(txts) - 1):
                # pattern predicted description
                patternDesc = txts.iloc[sim_sorted_doc_idx[-1][len(txts) - (i + 2)]]
                # pattern name matching the description
                patternName = (df["name"][(df["overview"] == patternDesc)]).to_string(
                    index=False
                )

                # percent cos sim match
                percentMatch = int(
                    (cos_sim[0][sim_sorted_doc_idx[-1][len(txts) - (i + 2)]]) * 100
                )

                # add the clustering by running it multiple times
                if patternName in PredictedPattern.allPatternsPredicted:
                    # print(patternName)
                    patternIndex = PredictedPattern.allPatternsPredicted.index(
                        patternName
                    )
                    # print(predictedPatterns[patternIndex].timesPredicted)

                    # print(patternIndex)
                    predictedPatterns[patternIndex].timesPredicted = (
                        predictedPatterns[patternIndex].timesPredicted + 1
                    )
                else:
                    predictedPatterns.append(
                        PredictedPattern(patternName, 1, percentMatch)
                    )
                    PredictedPattern.allPatternsPredicted.append(patternName)

        print("\n", algo, "\n")
        for predictedPattern in predictedPatterns:
            clusterPercent, cosSimPercent, totalPercent = CalculatePercent(
                predictedPattern,
                clusterWeight,
                cossimWeight,
                num_of_times_clustered,
            )
            predictedPattern.assignPercentages(clusterPercent, totalPercent)
        predictedPatterns = sorted(
            predictedPatterns, key=lambda x: x.totalPercent, reverse=True
        )
        k = 0
        for predictedPattern in predictedPatterns:
            k += 1
            print(
                "{}th pattern:  {:<25}{}%  CosSim match {}% Clustering Match {}% Total Match".format(
                    k,
                    predictedPattern.name,
                    predictedPattern.cosineSimPercent,
                    predictedPattern.clusterPercent,
                    predictedPattern.totalPercent,
                )
            )

    # clean up
    # Using drop() function to delete last row
    df.drop(index=n, axis=0, inplace=True)
    df = df.drop([str(algo)], axis=1)

    return output


if __name__ == "__main__":
    main()