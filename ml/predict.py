"""
Missing some NLTK data? Make sure you download 'punkt' and 'stopwords'.
`python -m nltk.downloader punkt stopwords`
"""

import re
import sys

import nltk
import nltk.corpus
import numpy as np
import pandas as pd
from fcmeans import FCM
from nltk import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn import cluster
from sklearn.cluster import AgglomerativeClustering, BisectingKMeans, KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, normalize
from sklearn_extra.cluster import KMedoids


def getTrainData(fileName):
    df = pd.read_csv(fileName)
    df = df.drop_duplicates(subset=["name"])

    return df


# removes a list of words (ie. stopwords) from a tokenized list.
def removeWords(listOfTokens, listOfWords):
    return [token for token in listOfTokens if token not in listOfWords]


# applies stemming to a list of tokenized words
def applyStemming(listOfTokens, stemmer):
    return [stemmer.stem(token) for token in listOfTokens]


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
        corpus[index] = corpus[index].replace('"', "")  # Removes double quotation marks
        corpus[index] = corpus[index].replace("'", "")  # Removes single quotation marks
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

    return corpus


# TODO: use k-means before this chunk of code to classify the problem with a
#  pattern class, then perform cosine similarity with the problem and the list
#  of candidate patterns from that class.
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


def displayPredictions(cos_sim, txts, df):
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


def runAlgorithms(final_df, df):
    # bisecting_strategy{“biggest_inertia”, “largest_cluster”}, default=”biggest_inertia”
    final_df_array = final_df.to_numpy()

    Bi_Bisect = BisectingKMeans(n_clusters=3, bisecting_strategy="biggest_inertia")
    Lc_Bisect = BisectingKMeans(n_clusters=3, bisecting_strategy="largest_cluster")
    Hierarchy = AgglomerativeClustering(n_clusters=3)
    Fuzzy_Means = FCM(n_clusters=3)
    Fuzzy_Means.fit(final_df_array)
    kmed = KMedoids(n_clusters=3)
    kmed_manhattan = KMedoids(n_clusters=3, metric="manhattan")
    Kmeans = cluster.KMeans(n_clusters=3)

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


def validateInput(designProblem):
    numOfWords = len(designProblem.split())
    return numOfWords >= 30 and numOfWords <= 120


def main():
    np.random.seed(9)

    print("Running program")
    dp_1 = sys.argv[1]

    # df = pd.read_csv('sourcemaking.csv')
    #
    # corpus_with_dp = pd.concat([df['text'],pd.Series(dp_1)],ignore_index=True)
    #
    # # display(corpus_with_dp.iloc[-1])
    #
    # vect = TfidfVectorizer(min_df=1, stop_words="english")
    # tfidf = vect.fit_transform(corpus_with_dp)
    # pairwise_similarity = tfidf * tfidf.T
    #
    # cos_sim_dp1 = pairwise_similarity.toarray()[-1].tolist()
    #
    # df1 = pd.DataFrame(data={'pattern':df['pattern_name'],'cos_sim':cos_sim_dp1[:-1],'sorted_indices':np.argsort(np.argsort(cos_sim_dp1))[:-1]})
    #
    # # print(df1)
    # from natsort import index_natsorted
    #
    # print(df1.sort_values(by='cos_sim',ascending=False)[:10].iloc[:,:2])

    language = "english"
    stemmer = PorterStemmer()
    vectorizer = TfidfVectorizer(sublinear_tf=True)
    df = getTrainData("GOF Patterns (2.0).csv")

    if not validateInput(dp_1):
        print("Invalid input size! please try again. \n")
        return

    problemRow = {"name": "design problem", "correct_category": 4, "overview": dp_1}
    df = df.append(problemRow, ignore_index=True)

    corpus = df["overview"].tolist()
    corpus = processCorpus(corpus, language, stemmer)

    X = vectorizer.fit_transform(corpus)
    tf_idf = pd.DataFrame(data=X.toarray(), columns=vectorizer.get_feature_names_out())

    runAlgorithms(tf_idf, df)

    cos_sim, txts = cosine_sim(df, df["overview"], df["Kmeans"].iloc[df.index[-1]], 1)
    displayPredictions(cos_sim, txts, df)

    # calculate the RCD
    # RCD = number of right design patterns / total suggested design patterns
    # This is a fraction of the suggested patterns that were in the correct cluster.
    rcd = 0
    if len(txts.loc[df["Kmeans"] == df["correct_category"]]) > 1:
        rcd = (len(txts.loc[df["Kmeans"] == df["correct_category"]]) - 1) / (len(txts) - 1)
    print("RCD = ", rcd)


if __name__ == "__main__":
    main()
