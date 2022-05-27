from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np


def compute(data, column_passage, column_query):
    # create tfidf of passages
    vectorizer_passages = TfidfVectorizer()
    vectorizer_passages.fit(data[column_passage])
    doc_vector_passages = vectorizer_passages.transform(data[column_passage])

    # tfidf to dataframe for further computation
    df_passages = pd.DataFrame(doc_vector_passages.toarray(
    ), columns=vectorizer_passages.get_feature_names())
    data = compute_LM(data, df_passages, column_query)
    data = compute_VSM(data, df_passages, column_query)
    data = compute_Jaccard(data, column_passage, column_query)
    return data


def compute_LM(data, df_passages, column_query):
    # laplace smoother value for 0 values
    la_place_smoothing = 1

    # compute probabilities for query and passage

    for line in range(len(df_passages)):
        token_words = data[column_query].loc[line].split(" ")
        tfidf_passage = df_passages.loc[line]
        prob = 1

        for word in token_words:
            if word in df_passages.columns:
                prob *= tfidf_passage[word] + la_place_smoothing
            else:
                prob *= la_place_smoothing
        prob /= len(tfidf_passage) + len(tfidf_passage) * la_place_smoothing
        data['LM'].loc[line] = prob

    return data


def compute_VSM(data, df_passages, column_query):
    for line in range(len(df_passages)):
        query_vector = []
        passage_vector = []
        query_words = data[column_query].loc[line].split(" ")
        for word in query_words:
            query_vector.append(1)
            try:
                passage_vector.append(df_passages.loc[line][word])
            except:
                passage_vector.append(0)
        qarray = np.array(query_vector).reshape(1, -1)
        parray = np.array(passage_vector).reshape(1, -1)
        cos = cosine_similarity(qarray, parray)
        data['VSM'].loc[line] = cos[0][0]
    return data

def compute_Jaccard(data, column_passage, column_query):

    data['Jaccard'] = data.apply(lambda row: jaccard_sim(row[column_passage],row[column_query]),axis=1)
    return data

def jaccard_sim(passage, query):
    intersection = set(query).intersection(set(passage))
    union = set(query).union(set(passage))
    return len(intersection) / len(union)

# path = r'C:\Users\timja\OneDrive\Dokumente\Master_Uni_Mannheim\Semester_3\Information_Retrieval\top1000.dev'

# data = pd.read_csv(path, sep='\t', header=None, names=['qid', 'pid', 'query', 'passage'])
# datashortend = data.loc[0:9999]
# datashortend['LM'] = 1
# df = compute(datashortend, 'passage', 'query')
# print(df.head())
