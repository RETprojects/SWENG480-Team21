import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


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