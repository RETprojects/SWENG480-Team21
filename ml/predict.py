# Command line usage: `python predict.py "Insert design problem here."`

import os
import sys

import nltk
import numpy as np
import pandas as pd
from fcmeans import FCM
from nltk import PorterStemmer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from sklearn import cluster
from sklearn.cluster import AgglomerativeClustering, BisectingKMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn_extra.cluster import KMedoids

# Read more here https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
pd.options.mode.chained_assignment = None

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

# This is a dictionary of all of the patterns listed in each GoF category.
patterns = {
    "creational": {
        "abstract_factory",
        "builder",
        "factory_method",
        "prototype",
        "singleton",
    },
    "structural": {
        "adapter",
        "bridge",
        "composite",
        "decorator",
        "facade",
        "flyweight",
        "proxy",
    },
    "behavioral": {
        "chain_of_responsibility",
        "command",
        "interpreter",
        "iterator",
        "mediator",
        "memento",
        "observer",
        "state",
        "strategy",
        "template_method",
        "visitor",
    },
}

stemmer = PorterStemmer()

# This is for the Django web app. This can definitely be improved.
output = []


def do_output(text: str = "") -> None:
    output.append(text)
    print(text)


# Modified from https://stackoverflow.com/a/20007730
def to_ordinal(n: int) -> str:
    if n < 1 or n > 9:
        raise ValueError
    return str(n) + ["st", "nd", "rd", "th"][min(n - 1, 3)]


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


def cosine_sim(df: pd.DataFrame, predicted_cluster: int) -> dict:
    cos_sim_dict = {}

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df["overview"])

    for algorithm in algorithms:
        indexes = df[df[algorithm] == predicted_cluster].index
        cos_sim_dict[algorithm] = cosine_similarity(
            X[indexes][:-1], X[indexes][-1]
        ).mean()

    return cos_sim_dict


# TODO: Recommend a pattern category in addition to patterns.
# Idea: if we find that a majority of the candidate patterns or the most
# recommended patterns for a problem belong to a category according to the
# correct_category label, then we can recommend that overall category for the
# design problem. Try Hussain et al. 2017 section 7.1, Pseudocode-2, but with
# clearly established categories for each design pattern involved.


def display_predictions(df: pd.DataFrame, cos_sim: np.int32) -> None:
    # Display the name of the pattern category corresponding to the most
    # recommended pattern.
    creational_count = 0
    behavioral_count = 0
    structural_count = 0

    for name in df["name"]:
        if name in patterns["creational"]:
            creational_count += 1
        elif name in patterns["structural"]:
            structural_count += 1
        elif name in patterns["behavioral"]:
            behavioral_count += 1

    if creational_count >= behavioral_count and creational_count >= structural_count:
        do_output("Category is most likely to be Creational.")
    elif behavioral_count >= creational_count and behavioral_count >= structural_count:
        do_output("Category is most likely to be Behavioral.")
    else:
        do_output("Category is most likely to be Structural.")
    do_output()

    # Show only the first 5 recommendations
    for row in df[:5].itertuples():
        friendly_index = to_ordinal(row.Index + 1)
        percent_match = int(round(cos_sim, 2) * 100)

        do_output(
            f"{friendly_index} pattern: {row.name} {percent_match}% match",
        )
    do_output()


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
    fcm = FCM(n_clusters=3, random_state=9)
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


def main(design_problem: str = ""):
    # Handle command line execution
    if not design_problem:
        design_problem = sys.argv[1]

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

    # Final example demonstrates how to append a Series as a row
    # https://pandas.pydata.org/docs/reference/api/pandas.concat.html
    new_row = pd.Series(
        {
            "name": "design_problem",
            "overview": design_problem,
        }
    )
    df = pd.concat([df, new_row.to_frame().T], ignore_index=True)

    # Preprocess
    cleaned_text = preprocess(df["overview"])
    # Create a dense tfidf matrix
    tfidf_matrix = do_weighting("Tfidf", cleaned_text)
    # Perform clustering
    df_labels = do_cluster(tfidf_matrix)
    # Append (horizontally) the cluster labels to the original DF
    df = pd.concat([df, df_labels], axis=1)

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df["overview"])

    max_len = len(max(algorithms_pretty, key=len))
    do_output()
    for i, algorithm in enumerate(algorithms):
        do_output(f"{algorithms_pretty[i]}")
        do_output("-" * max_len)

        predicted_cluster = df[algorithm].iloc[-1]

        df_problem_cluster = df.loc[df[algorithm] == predicted_cluster]
        indexes = df_problem_cluster.index
        cos_sim = cosine_similarity(X[indexes], X[indexes][-1]).flatten()

        df_problem_cluster["match"] = cos_sim

        # Calculate the most likely category
        predicted_category = (
            df_problem_cluster.iloc[:-1][["category", "match"]]
            .groupby("category")["match"]
            .mean()
            .idxmax()
        )
        do_output(f"Category is most likely {predicted_category}\n")

        for name, percent in sorted(
            list(zip(df_problem_cluster.name, cos_sim))[:-1],
            key=lambda x: x[1],
            reverse=True,
        ):
            do_output(f"{name} {percent:.0%}")
        do_output()

    return output


if __name__ == "__main__":
    main()
