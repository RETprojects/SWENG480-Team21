import sys

from fcmeans                          import FCM

# Data Structures
import numpy  as np
import pandas as pd
import json
# Corpus Processing
import re
import nltk
import nltk.corpus
from nltk.tokenize                    import word_tokenize
from nltk.stem                        import WordNetLemmatizer
from nltk                             import SnowballStemmer, PorterStemmer
nltk.download('punkt')

from sklearn.feature_extraction.text  import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing            import normalize, Normalizer
from sklearn.decomposition            import PCA, TruncatedSVD
from sklearn.cluster                  import KMeans, BisectingKMeans, AgglomerativeClustering
from sklearn_extra.cluster            import KMedoids
from sklearn.pipeline                 import make_pipeline

from unidecode                        import unidecode

# K-Means
from sklearn                          import cluster

# Visualization and Analysis
import matplotlib.pyplot  as plt
import matplotlib.cm      as cm
import seaborn            as sns
from sklearn.metrics                  import silhouette_samples, silhouette_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
from wordcloud                        import WordCloud

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity

np.random.seed(9)


# some more functions

def getTrainData(fileName):
  df = pd.read_csv(fileName)
  df = df.drop_duplicates(subset=['name'])

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


nltk.download('stopwords')


def processCorpus(corpus, language, stemmer):
    stopwords = nltk.corpus.stopwords.words(language)
    param_stemmer = stemmer

    for document in corpus:
        index = corpus.index(document)
        corpus[index] = str(corpus[index]).replace(u'\ufffd', '8')  # Replaces the ASCII '�' symbol with '8'
        corpus[index] = corpus[index].replace('"', '')  # Removes double quotation marks
        corpus[index] = corpus[index].replace("'", "")  # Removes single quotation marks
        corpus[index] = corpus[index].replace(',', '')  # Removes commas
        corpus[index] = corpus[index].rstrip('\n')  # Removes line breaks
        corpus[index] = corpus[index].casefold()  # Makes all letters lowercase

        corpus[index] = re.sub('\W_', ' ', corpus[index])  # removes specials characters and leaves only words
        corpus[index] = re.sub("\S*\d\S*", " ", corpus[
            index])  # removes numbers and words concatenated with numbers IE h4ck3r. Removes road names such as BR-381.
        corpus[index] = re.sub("\S*@\S*\s?", " ", corpus[index])  # removes emails and mentions (words with @)
        corpus[index] = re.sub(r'http\S+', '', corpus[index])  # removes URLs with http
        corpus[index] = re.sub(r'www\S+', '', corpus[index])  # removes URLs with www

        listOfTokens = word_tokenize(corpus[index])
        twoLetterWord = twoLetters(listOfTokens)

        listOfTokens = removeWords(listOfTokens, stopwords)
        listOfTokens = removeWords(listOfTokens, twoLetterWord)

        listOfTokens = applyStemming(listOfTokens, param_stemmer)

        corpus[index] = " ".join(listOfTokens)
        corpus[index] = unidecode(corpus[index])

    return corpus


# TODO: use k-means before this chunk of code to classify the problem with a pattern class, then perform cosine similarity with the problem and the list of candidate patterns from that class.
# Source: https://danielcaraway.github.io/html/sklearn_cosine_similarity.html

def cosine_sim(df, df_col, class_no, pos_to_last):
    unigram_count = CountVectorizer(encoding='latin-1', binary=False)
    unigram_count_stop_remove = CountVectorizer(encoding='latin-1', binary=False, stop_words='english')

    # get the list of candidate patterns
    txts = df_col.loc[df['Kmeans'] == class_no]  # where label == class_no
    vecs = unigram_count.fit_transform(txts)

    cos_sim = cosine_similarity(vecs[-pos_to_last], vecs)
    # sim_sorted_doc_idx = cos_sim.argsort()
    # print the most similar pattern to the problem; it's actually the problem itself
    # print("Design Problem: \n" + txts.iloc[sim_sorted_doc_idx[-1][len(txts)-1]] + "\n")

    # bestFittingPatternDesc = txts.iloc[sim_sorted_doc_idx[-1][len(txts)-2]]

    # print the second most similar pattern; it's likely the best-fitting design pattern for the design problem
    # print(txts[sim_sorted_doc_idx[-1][len(txts)-2]])
    # print("\nCorrect Pattern: " + (df['name'][(df['overview'] == bestFittingPatternDesc)]).to_string(index=False) + "\n")

    return cos_sim, txts

def displayPredictions(cos_sim, txts, df):
  sim_sorted_doc_idx = cos_sim.argsort()
  for i in range(len(txts) - 1):
    patternDesc = txts.iloc[sim_sorted_doc_idx[-1][len(txts)-(i + 2)]]
    patternName = (df['name'][(df['overview'] == patternDesc)]).to_string(index=False)
    percentMatch = int((cos_sim[0][sim_sorted_doc_idx[-1][len(txts)-(i + 2)]]) * 100)
    print("{}th pattern:  {:<20}{}%  match".format(i+1, patternName, percentMatch))

def runAlgorithms(final_df, df):
  # bisecting_strategy{“biggest_inertia”, “largest_cluster”}, default=”biggest_inertia”
  final_df_array = final_df.to_numpy()

  Bi_Bisect = BisectingKMeans(n_clusters=3, bisecting_strategy="biggest_inertia")
  Lc_Bisect = BisectingKMeans(n_clusters=3, bisecting_strategy="largest_cluster")
  Hierarchy = AgglomerativeClustering(n_clusters=3)
  Fuzzy_Means = FCM(n_clusters=3)
  Fuzzy_Means.fit(final_df_array)
  kmed = KMedoids(n_clusters=3)
  kmed_manhattan = KMedoids(n_clusters=3,metric='manhattan')
  Kmeans = cluster.KMeans(n_clusters = 3)

  Kmeans_labels = Kmeans.fit_predict(final_df)
  fuzzy_labels = Fuzzy_Means.predict(final_df_array)
  bi_bisect_labels = Bi_Bisect.fit_predict(final_df)
  lc_bisect_labels = Lc_Bisect.fit_predict(final_df)
  hierarchy_labels = Hierarchy.fit_predict(final_df)
  kmed_labels = kmed.fit_predict(final_df)
  kmed_man_labels = kmed_manhattan.fit_predict(final_df)

  df['Kmeans'] = Kmeans_labels
  df['fuzzy'] = fuzzy_labels
  df['hierarchy'] = hierarchy_labels
  df['Bi_Bisect'] = bi_bisect_labels
  df['Lc_Bisect'] = lc_bisect_labels
  df['PAM-EUCLIDEAN'] = kmed_labels
  df['PAM-MANHATTAN'] = kmed_man_labels

def validateInput(designProblem):
    numOfWords = len(designProblem.split())
    if (numOfWords < 30 or numOfWords > 120):
        return False
    return True

def main():
    print("Running program")
    dp_1 = sys.argv[1]

    df = pd.read_csv('sourcemaking.csv')

    corpus_with_dp = pd.concat([df['text'],pd.Series(dp_1)],ignore_index=True)

    # display(corpus_with_dp.iloc[-1])

    vect = TfidfVectorizer(min_df=1, stop_words="english")                                                                                                                                                                                                   
    tfidf = vect.fit_transform(corpus_with_dp)                                                                                                                                                                                                                       
    pairwise_similarity = tfidf * tfidf.T 

    cos_sim_dp1 = pairwise_similarity.toarray()[-1].tolist()

    df1 = pd.DataFrame(data={'pattern':df['pattern_name'],'cos_sim':cos_sim_dp1[:-1],'sorted_indices':np.argsort(np.argsort(cos_sim_dp1))[:-1]})

    # print(df1)
    from natsort import index_natsorted

    print(df1.sort_values(by='cos_sim',ascending=False)[:10].iloc[:,:2])

if __name__ == "__main__":
    main()