"""
Command line usage: `python predict.py "Insert design problem here."`

Missing some NLTK data? Make sure you download 'punkt' and 'stopwords'.
`python -m nltk.downloader punkt stopwords`
"""

import os
import re
import sys

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
from sklearn.metrics import silhouette_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn_extra.cluster import KMedoids

# global variables
algos = [
    "kmeans",
    "fuzzy_cmeans",
    "agglomerative",
    "bi_kmeans_inertia",
    "bi_kmeans_lg_cluster",
    "pam_euclidean",
    "pam_manhattan",
]


def preprocess(corpus):
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


# TODO: Recommend a pattern category in addition to patterns.


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


# Functions from PatternKMeans


def Silhouette(vector_data, cluster_labels):
    # range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9]
    # silhouette_avg = []
    # for num_clusters in range_n_clusters:
    #   # initialise kmeans
    #   kmeans = KMeans(n_clusters=num_clusters, n_init='auto')
    #   kmeans.fit(vector_data)
    #   cluster_labels = kmeans.labels_

    # silhouette score
    # silhouette_avg.append(silhouette_score(vector_data, cluster_labels))
    s_avg = silhouette_score(vector_data, cluster_labels)
    return s_avg
    # plt.plot(range_n_clusters,silhouette_avg,'bx-')

    # plt.xlabel('Values of K')
    # plt.ylabel('Silhouette score')
    # plt.title('Silhouette analysis For Optimal k')
    # plt.show()


def getFScore(labels, df):
    df2 = df.pivot_table(index=["correct_category"], aggfunc="size")

    num_of_creational = df2[2]
    num_of_structural = df2[1]
    num_of_behavioral = df2[0]

    true_1 = [0] * num_of_creational + [1] * num_of_structural + [2] * num_of_behavioral
    true_2 = [0] * num_of_creational + [2] * num_of_structural + [1] * num_of_behavioral
    true_3 = [1] * num_of_creational + [0] * num_of_structural + [2] * num_of_behavioral
    true_4 = [1] * num_of_creational + [2] * num_of_structural + [0] * num_of_behavioral
    true_5 = [2] * num_of_creational + [0] * num_of_structural + [1] * num_of_behavioral
    true_6 = [2] * num_of_creational + [1] * num_of_structural + [0] * num_of_behavioral

    # print('===========KMEANS===========')
    # print('Predicted labels:')
    # display(Kmeans_labels.tolist())

    fscores = [
        f1_score(true_1, labels.tolist(), average="micro"),
        f1_score(true_2, labels.tolist(), average="micro"),
        f1_score(true_3, labels.tolist(), average="micro"),
        f1_score(true_4, labels.tolist(), average="micro"),
        f1_score(true_5, labels.tolist(), average="micro"),
        f1_score(true_6, labels.tolist(), average="micro"),
    ]

    km_best = np.around(max(fscores), 3)
    # print('\nBest fscore is:', km_best, 'from true_' + str(np.argmax(fscores) + 1))
    # display(globals()['true_' + str(np.argmax(fscores) + 1)])
    return km_best


# TODO: put this somewhere

# kmeans_fscore_avg.append(getFScore(Kmeans_labels, df))
# kmeans_silhouette_avg.append(Silhouette(tfidf, Kmeans_labels))
#
# fmeans_fscore_avg.append(getFScore(fuzzy_labels, df))
# fmeans_silhouette_avg.append(Silhouette(tfidf, fuzzy_labels))
#
# hier_fscore_avg.append(getFScore(hierarchy_labels, df))
# hier_silhouette_avg.append(Silhouette(tfidf, hierarchy_labels))

# kmed_fscore_avg.append(getFScore(kmed_labels, df))
# kmed_silhouette_avg.append(Silhouette(tfidf, kmed_labels))

# kmed_man_fscore_avg.append(getFScore(kmed_man_labels, df))
# kmed_man_silhouette_avg.append(Silhouette(tfidf, kmed_man_labels))

# bi_bisect_fscore_avg.append(getFScore(bi_bisect_labels, df))
# bi_bisect_silhouette_avg.append(Silhouette(tfidf, bi_bisect_labels))

# lc_bisect_fscore_avg.append(getFScore(lc_bisect_labels, df))
# lc_bisect_silhouette_avg.append(Silhouette(tfidf, lc_bisect_labels))


def main():
    kmeans_silhouette_avg = []
    kmeans_fscore_avg = []

    fmeans_silhouette_avg = []
    fmeans_fscore_avg = []

    hier_silhouette_avg = []
    hier_fscore_avg = []

    kmed_silhouette_avg = []
    kmed_fscore_avg = []

    kmed_man_silhouette_avg = []
    kmed_man_fscore_avg = []

    bi_bisect_silhouette_avg = []
    bi_bisect_fscore_avg = []

    lc_bisect_silhouette_avg = []
    lc_bisect_fscore_avg = []

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

    for a_name in algos:
        print("---------", a_name, "------------")

        CosSimDict, TxtsDict = cosine_sim(
            df, df["overview"], df[a_name].iloc[df.index[-1]], 1
        )
        cos_sim = CosSimDict[a_name]
        txts = TxtsDict[a_name]
        display_predictions(cos_sim, txts, df)

        # Calculate the RCD
        # RCD = number of right design patterns / total suggested design patterns
        # This is a fraction of the suggested patterns that were in the correct cluster.
        # TODO: We probably need to account for the fact that cluster labels
        #       may not be the same every run (0 isn't always behavioral,
        #       for example). Jonathan may have already accounted for this
        #       with the getFScore function.
        rcd = 0
        if len(txts.loc[df[a_name] == df["correct_category"]]) > 1:
            rcd = (len(txts.loc[df[a_name] == df["correct_category"]]) - 1) / (
                len(txts) - 1
            )
        print("RCD = ", rcd)


if __name__ == "__main__":
    main()
