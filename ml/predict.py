# Command line usage: `python predict.py "Insert design problem here."`

import os
import sys

import nltk
import numpy as np
import pandas as pd
from fcmeans import FCM
from matplotlib import pyplot as plt
from matplotlib import cm as cm
from nltk import PorterStemmer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from sklearn import cluster
from sklearn.cluster import AgglomerativeClustering, BisectingKMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_samples, silhouette_score
from wordcloud import WordCloud

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


# from PatternKMeans (last edited Jan 2, 2023)
# Transforms a centroids dataframe into a dictionary to be used on a WordCloud.
def centroidsDict(centroids, index):
    a = centroids.T[index].sort_values(ascending=False).reset_index().values
    centroid_dict = dict()

    for i in range(0, len(a)):
        centroid_dict.update({a[i, 0]: a[i, 1]})

    return centroid_dict


# Generates a word cloud of the most frequent and influential words in a cluster.
def generateWordClouds(centroids):
    wordcloud = WordCloud(max_font_size=100, background_color="white")
    for i in range(0, len(centroids)):
        centroid_dict = centroidsDict(centroids, i)
        wordcloud.generate_from_frequencies(centroid_dict)

        plt.figure()
        plt.title("Cluster {}".format(i))
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.show()


def run_KMeans(max_k, data):
    max_k += 1
    kmeans_results = dict()
    for k in range(2, max_k):
        kmeans = cluster.KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            tol=0.0001
            # , n_jobs = -1
            ,
            random_state=1,
            algorithm="full",
        )

        kmeans_results.update({k: kmeans.fit(data)})

    return kmeans_results


def printAvg(avg_dict):
    for avg in sorted(avg_dict.keys(), reverse=True):
        print("Avg: {}\tK:{}".format(avg.round(4), avg_dict[avg]))


def plotSilhouette(df, n_clusters, kmeans_labels, silhouette_avg):
    fig, ax1 = plt.subplots(1)
    fig.set_size_inches(8, 6)
    ax1.set_xlim([-0.2, 1])
    ax1.set_ylim([0, len(df) + (n_clusters + 1) * 10])

    ax1.axvline(
        x=silhouette_avg, color="red", linestyle="--"
    )  # The vertical line for average silhouette score of all the values
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.title(
        ("Silhouette analysis for K = %d" % n_clusters), fontsize=10, fontweight="bold"
    )

    y_lower = 10
    sample_silhouette_values = silhouette_samples(
        df, kmeans_labels
    )  # Compute the silhouette scores for each sample
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[kmeans_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        ax1.text(
            -0.05, y_lower + 0.5 * size_cluster_i, str(i)
        )  # Label the silhouette plots with their cluster numbers at the middle
        y_lower = (
            y_upper + 10
        )  # Compute the new y_lower for next plot. 10 for the 0 samples
    plt.show()


def silhouette(kmeans_dict, df, plot=False):
    df = df.to_numpy()
    avg_dict = dict()
    for n_clusters, kmeans in kmeans_dict.items():
        kmeans_labels = kmeans.predict(df)
        silhouette_avg = silhouette_score(
            df, kmeans_labels
        )  # Average Score for all Samples
        avg_dict.update({silhouette_avg: n_clusters})

        if plot:
            plotSilhouette(df, n_clusters, kmeans_labels, silhouette_avg)


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


# TODO: After clustering the patterns, automatically label each cluster.
#       Each cluster can have a word cloud, and the clusters can be labeled
#       using the most important words in each cluster as a guide.
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


def main(design_problem: str = ""):
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

    # Preprocess
    cleaned_text = preprocess(df["overview"])
    # Create a dense tfidf matrix
    tfidf_matrix = do_weighting("Tfidf", cleaned_text)
    # Perform clustering
    df_labels = do_cluster(tfidf_matrix)
    # Append (horizontally) the cluster labels to the original DF
    df = pd.concat([df, df_labels], axis=1)

    # Used for cosine similarity
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df["overview"])

    # Used for output formatting
    max_len_pattern = df["name"].str.len().max()
    max_len = len(max(algorithms_pretty, key=len))

    do_output()
    for i, algorithm in enumerate(algorithms):
        do_output(f"{algorithms_pretty[i]}")
        do_output("-" * max_len)

        # Get the cluster number of the design problem
        predicted_cluster = df[algorithm].iloc[-1]

        """
        1. Filter all rows where the cluster number matches
        2. Select the name, category, and `algorithm` rows
        3. Remove the design problem row
        4. Make a copy to make the logic more clear
        """
        df_problem_cluster = (
            df.loc[df[algorithm] == predicted_cluster][["name", "category", algorithm]]
            .iloc[:-1]
            .copy()
        )

        # Calculate cosine similarity for all patterns in the cluster vs. design problem
        df_problem_cluster["match"] = cosine_similarity(
            X[df_problem_cluster.index], X[-1]
        ).flatten()

        # Calculate the most likely category
        predicted_category = (
            df_problem_cluster.groupby("category")["match"].mean().idxmax()
        )
        do_output(f"Category is most likely {predicted_category}\n")

        # Display the matching patterns by match % in descending order
        for index, (name, percent) in enumerate(
            sorted(
                zip(df_problem_cluster["name"], df_problem_cluster["match"]),
                key=lambda x: x[1],
                reverse=True,
            )
        ):
            do_output(
                f"{ordinal(index + 1).ljust(4)} pattern: {name.ljust(max_len_pattern)} {percent:.0%}"
            )
        do_output()

    return output


if __name__ == "__main__":
    main()
