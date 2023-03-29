# Command line usage: `python predict.py "Insert design problem here."`

import os
import re
import sys

import nltk
import numpy as np
import pandas as pd
from fcmeans import FCM
from nltk import PorterStemmer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from sklearn import cluster
from sklearn.cluster import AgglomerativeClustering, BisectingKMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn_extra.cluster import KMedoids

try:
    nltk.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

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


def preprocess(corpus: list) -> list:
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
            [
                stemmer.stem(word)
                for word in document.split()
                if (
                    word not in stop_words
                    and (
                        pos_tag(word_tokenize(word), tagset="universal")[0][1] == "VERB"
                        or pos_tag(word_tokenize(word), tagset="universal")[0][1]
                        == "ADJ"
                    )
                )
            ]
        ).strip()

        # Replace the original string
        corpus[index] = document

    return corpus


# Source: https://danielcaraway.github.io/html/sklearn_cosine_similarity.html
def cosine_sim(df: pd.DataFrame, predicted_cluster: int) -> tuple[dict, dict]:
    unigram_count = CountVectorizer()

    # Loop through the clustering algorithms and calculate the cosine similarity measures based on each algorithm.
    cos_sim_dict = {}
    txts_dict = {}
    for algorithm in algorithms:
        # get the list of candidate patterns
        txts = df["overview"].loc[
            df[algorithm] == predicted_cluster
        ]  # where label == predicted_cluster
        vecs = unigram_count.fit_transform(txts)

        cos_sim = cosine_similarity(vecs[-1], vecs)

        # add cos_sim and txts to the dictionaries with the algorithm name as the key
        cos_sim_dict[algorithm] = cos_sim
        txts_dict[algorithm] = txts

    # return cos_sim, txts
    return cos_sim_dict, txts_dict


# TODO: Recommend a pattern category in addition to patterns.
# Idea: if we find that a majority of the candidate patterns or the most
# recommended patterns for a problem belong to a category according to the
# correct_category label, then we can recommend that overall category for the
# design problem. Try Hussain et al. 2017 section 7.1, Pseudocode-2, but with
# clearly established categories for each design pattern involved.


def display_predictions(cos_sim: np.ndarray, txts: pd.Series, df: pd.DataFrame) -> None:
    sim_sorted_doc_idx = cos_sim.argsort()
    for i in range(len(txts) - 1):
        pattern_desc = txts.iloc[sim_sorted_doc_idx[-1][len(txts) - (i + 2)]]
        pattern_name = (df["name"][(df["overview"] == pattern_desc)]).to_string(
            index=False
        )
        percentMatch = int(
            (cos_sim[0][sim_sorted_doc_idx[-1][len(txts) - (i + 2)]]) * 100
        )
        print(
            "{}th pattern:  {:<20}{}%  match".format(i + 1, pattern_name, percentMatch)
        )

    # Display the name of the pattern category corresponding to the most
    # recommended pattern.
    top_pattern_desc = txts.iloc[sim_sorted_doc_idx[-1][len(txts) - 2]]
    # top_pattern_name = (df["name"][(df["overview"] == top_pattern_desc)]).to_string(
    #     index=False
    # )
    top_pattern_cat_num = df.loc[
        df["overview"] == top_pattern_desc, "correct_category"
    ].iloc[0]
    top_pattern_cat_name = ""
    if top_pattern_cat_num == 0:
        top_pattern_cat_name = "Behavioral (GoF)"
    elif top_pattern_cat_num == 1:
        top_pattern_cat_name = "Structural (GoF)"
    else:
        top_pattern_cat_name = "Creational (GoF)"
    print("Most recommended pattern: ", top_pattern_cat_name)


def do_cluster(df_weighted: pd.DataFrame) -> pd.DataFrame:
    # This is the DataFrame we will return that contains all the labels.
    df = pd.DataFrame()

    # Agglomerative (hierarchical)
    agg = AgglomerativeClustering(n_clusters=3)
    df["agglomerative"] = agg.fit_predict(df_weighted)

    # Bisecting k-means
    bisect_inertia = BisectingKMeans(n_clusters=3)
    bisect_lg_cluster = BisectingKMeans(
        n_clusters=3, bisecting_strategy="largest_cluster"
    )
    df["bi_kmeans_inertia"] = bisect_inertia.fit_predict(df_weighted)
    df["bi_kmeans_lg_cluster"] = bisect_lg_cluster.fit_predict(df_weighted)

    # Fuzzy c-means
    final_df_np = df_weighted.to_numpy()
    fcm = FCM(n_clusters=3)
    fcm.fit(final_df_np)
    df["fuzzy_cmeans"] = fcm.predict(final_df_np)

    # K-means
    km = cluster.KMeans(n_clusters=3, n_init=10, random_state=9)
    df["kmeans"] = km.fit_predict(df_weighted)

    # K-medoids
    kmed_euclidean = KMedoids(n_clusters=3)
    kmed_manhattan = KMedoids(n_clusters=3, metric="manhattan")
    df["pam_euclidean"] = kmed_euclidean.fit_predict(df_weighted)
    df["pam_manhattan"] = kmed_manhattan.fit_predict(df_weighted)

    return df


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
    corpus = preprocess(corpus)

    # Add the predicted labels to the DataFrame (concat horizontally)
    df_labels = do_cluster(do_weighting("Tfidf", corpus))
    df = pd.concat([df, df_labels], axis=1)

    for algorithm in algorithms:
        print("---------", algorithm, "------------")

        cos_sim_dict, txts_dict = cosine_sim(df, df[algorithm].iloc[df.index[-1]])
        cos_sim = cos_sim_dict[algorithm]
        txts = txts_dict[algorithm]
        display_predictions(cos_sim, txts, df)

        # Calculate the RCD
        # RCD = number of right design patterns / total suggested design patterns
        # This is a fraction of the suggested patterns that were in the correct cluster.
        # TODO: We probably need to account for the fact that cluster labels
        #       may not be the same every run (0 isn't always behavioral,
        #       for example). Jonathan may have already accounted for this
        #       with the getFScore function.
        rcd = 0
        if len(txts.loc[df[algorithm] == df["correct_category"]]) > 1:
            rcd = (len(txts.loc[df[algorithm] == df["correct_category"]]) - 1) / (
                len(txts) - 1
            )
        print("RCD = ", rcd)


if __name__ == "__main__":
    main()
