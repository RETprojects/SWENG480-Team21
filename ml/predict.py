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

    # get the list of candidate patterns
    txts = df_col.loc[df["Kmeans"] == class_no]  # where label == class_no
    vecs = unigram_count.fit_transform(txts)

    cos_sim = cosine_similarity(vecs[-pos_to_last], vecs)

    return cos_sim, txts


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


def run_algorithms(final_df, df):
    final_df_array = final_df.to_numpy()

    Bi_Bisect = BisectingKMeans(n_clusters=3, bisecting_strategy="biggest_inertia")
    Lc_Bisect = BisectingKMeans(n_clusters=3, bisecting_strategy="largest_cluster")
    Hierarchy = AgglomerativeClustering(n_clusters=3)
    Fuzzy_Means = FCM(n_clusters=3)
    Fuzzy_Means.fit(final_df_array)
    kmed = KMedoids(n_clusters=3)
    kmed_manhattan = KMedoids(n_clusters=3, metric="manhattan")
    Kmeans = cluster.KMeans(n_clusters=3, n_init=10, random_state=9)

    Kmeans_labels = Kmeans.fit_predict(final_df)
    fuzzy_labels = Fuzzy_Means.predict(final_df_array)
    bi_bisect_labels = Bi_Bisect.fit_predict(final_df)
    lc_bisect_labels = Lc_Bisect.fit_predict(final_df)
    hierarchy_labels = Hierarchy.fit_predict(final_df)
    kmed_labels = kmed.fit_predict(final_df)
    kmed_man_labels = kmed_manhattan.fit_predict(final_df)

    df["Kmeans"] = Kmeans_labels
    df["fuzzy"] = fuzzy_labels
    df["hierarchy"] = hierarchy_labels
    df["Bi_Bisect"] = bi_bisect_labels
    df["Lc_Bisect"] = lc_bisect_labels
    df["PAM-EUCLIDEAN"] = kmed_labels
    df["PAM-MANHATTAN"] = kmed_man_labels


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

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    tf_idf = pd.DataFrame(data=X.toarray(), columns=vectorizer.get_feature_names_out())

    run_algorithms(tf_idf, df)

    cos_sim, txts = cosine_sim(df, df["overview"], df["Kmeans"].iloc[df.index[-1]], 1)
    display_predictions(cos_sim, txts, df)

    # calculate the RCD
    # RCD = number of right design patterns / total suggested design patterns
    # This is a fraction of the suggested patterns that were in the correct cluster.
    rcd = 0
    if len(txts.loc[df["Kmeans"] == df["correct_category"]]) > 1:
        rcd = (len(txts.loc[df["Kmeans"] == df["correct_category"]]) - 1) / (
            len(txts) - 1
        )
    print("RCD = ", rcd)


if __name__ == "__main__":
    main()
