"""
Command line usage: `python predict.py "Insert design problem here."`

Missing some NLTK data? Make sure you download 'punkt' and 'stopwords'.
`python -m nltk.downloader punkt stopwords`
"""

import os
import re
import sys

import pandas as pd
from fcmeans import FCM
from nltk import PorterStemmer
from nltk.corpus import stopwords
from sklearn import cluster
from sklearn.cluster import AgglomerativeClustering, BisectingKMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn_extra.cluster import KMedoids

# global variables
algos = [
    "Kmeans",
    "fuzzy",
    "hierarchy",
    "Bi_Bisect",
    "Lc_Bisect",
    "PAM-EUCLIDEAN",
    "PAM-MANHATTAN",
]


def process_corpus(corpus):
    # We use a set for better lookup performance
    stop_words = set(stopwords.words("english"))

    stemmer = PorterStemmer()

    for index, document in enumerate(corpus):
        # Lowercase the string
        document = document.lower()

        # Replace all non-alphabetical characters with whitespace, then compress duplicate whitespace
        document = re.sub("[^A-Za-z]", " ", document)
        document = re.sub("\s{2,}", " ", document)

        # Remove stop words, stem, and remove leading and trailing whitespace
        document = " ".join(
            [stemmer.stem(word) for word in document.split() if word not in stop_words]
        ).strip()

        # Replace the original string
        corpus[index] = document

    return corpus


# TODO: calculate cosine similarity measures for all clustering algorithms,
#       not just k-means
# Source: https://danielcaraway.github.io/html/sklearn_cosine_similarity.html
def cosine_sim(df, df_col, class_no, pos_to_last):
    unigram_count = CountVectorizer(encoding="latin-1", binary=False)
    unigram_count_stop_remove = CountVectorizer(
        encoding="latin-1", binary=False, stop_words="english"
    )

    # Loop through the clustering algorithms and calculate the cosine similarity measures based on each algorithm.
    CosSimDict = {}
    TxtsDict = {}
    for algo_name in algos:
        # get the list of candidate patterns
        txts = df_col.loc[df[algo_name] == class_no]  # where label == class_no
        vecs = unigram_count.fit_transform(txts)

        cos_sim = cosine_similarity(vecs[-pos_to_last], vecs)

        # add cos_sim and txts to the dictionaries with the algorithm name as the key
        CosSimDict[algo_name] = cos_sim
        TxtsDict[algo_name] = txts

    # return cos_sim, txts
    return CosSimDict, TxtsDict


def display_predictions(cos_sim, txts, df):
    sim_sorted_doc_idx = cos_sim.argsort()
    for i in range(len(txts) - 1):
        patternDesc = txts.iloc[sim_sorted_doc_idx[-1][len(txts) - (i + 2)]]
        patternName = (df["name"][(df["overview"] == patternDesc)]).to_string(
            index=False
        )
        percentMatch = int(
            (cos_sim[0][sim_sorted_doc_idx[-1][len(txts) - (i + 2)]]) * 100
        )
        print(
            "{}th pattern:  {:<20}{}%  match".format(i + 1, patternName, percentMatch)
        )


def run_algorithms(final_df: pd.DataFrame, df: pd.DataFrame) -> None:
    # Agglomerative (hierarchical)
    agg = AgglomerativeClustering(n_clusters=3)
    df["hierarchy"] = agg.fit_predict(final_df)

    # Bisecting k-means
    bisect = BisectingKMeans(n_clusters=3)
    bisect_lg_cluster = BisectingKMeans(
        n_clusters=3, bisecting_strategy="largest_cluster"
    )
    df["Bi_Bisect"] = bisect.fit_predict(final_df)
    df["Lc_Bisect"] = bisect_lg_cluster.fit_predict(final_df)

    # Fuzzy c-means
    final_df_np = final_df.to_numpy()
    fcm = FCM(n_clusters=3)
    fcm.fit(final_df_np)
    df["fuzzy"] = fcm.predict(final_df_np)

    # K-means
    km = cluster.KMeans(n_clusters=3, n_init=10, random_state=9)
    df["Kmeans"] = km.fit_predict(final_df)

    # K-medoids
    kmed = KMedoids(n_clusters=3)
    kmed_manhattan = KMedoids(n_clusters=3, metric="manhattan")
    df["PAM-EUCLIDEAN"] = kmed.fit_predict(final_df)
    df["PAM-MANHATTAN"] = kmed_manhattan.fit_predict(final_df)


# Better for this to be an enum, but the syntax is a bit tricky.
weighting_methods = {"Binary", "Count", "Tfidf"}


# Output: DataFrame with dense values
def do_weighting(method: str, series: pd.Series) -> pd.DataFrame:
    if method == "Binary":
        vectorizer = CountVectorizer(binary=True)
    elif method == "Count":
        vectorizer = CountVectorizer()
    elif method == "Tfidf":
        vectorizer = TfidfVectorizer()
    else:
        print("Error. Did not pass valid weighting method")
        return

    matrix = vectorizer.fit_transform(series)
    return pd.DataFrame.sparse.from_spmatrix(
        matrix, columns=vectorizer.get_feature_names_out()
    ).sparse.to_dense()


def main():
    # Load the data we are working with
    FILENAME = "GOF Patterns (2.0).csv"
    file_path = os.path.join(os.path.dirname(__file__), f"data/{FILENAME}")

    if FILENAME.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif FILENAME.endswith(".xls") or FILENAME.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        print("Unknown file extension. Ending program.")
        return

    design_problem = sys.argv[1]

    # Final example demonstrates how to append a Series as a row
    # https://pandas.pydata.org/docs/reference/api/pandas.concat.html
    new_row = (
        pd.Series(
            {
                "name": "design problem",
                "correct_category": 4,
                "overview": design_problem,
            }
        )
        .to_frame()
        .T
    )
    df = pd.concat([df, new_row], ignore_index=True)

    corpus = df["overview"].tolist()
    corpus = process_corpus(corpus)

    run_algorithms(do_weighting("Tfidf", corpus), df)

    for a_name in algos:
        print("---------", a_name, "------------")

        # cos_sim, txts = cosine_sim(df, df["overview"], df["Kmeans"].iloc[df.index[-1]], 1)
        CosSimDict, TxtsDict = cosine_sim(
            df, df["overview"], df[a_name].iloc[df.index[-1]], 1
        )
        cos_sim = CosSimDict[a_name]
        txts = TxtsDict[a_name]
        display_predictions(cos_sim, txts, df)

        # calculate the RCD
        # RCD = number of right design patterns / total suggested design patterns
        # This is a fraction of the suggested patterns that were in the correct cluster.
        rcd = 0
        if len(txts.loc[df[a_name] == df["correct_category"]]) > 1:
            rcd = (len(txts.loc[df[a_name] == df["correct_category"]]) - 1) / (
                len(txts) - 1
            )
        print("RCD = ", rcd)


if __name__ == "__main__":
    main()
