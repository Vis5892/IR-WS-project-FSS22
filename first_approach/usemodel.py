import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from compute_LM_VSM import compute
from sklearn.metrics import average_precision_score
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import tarfile
import gzip
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string

def calculateMRR10(list, id):
    result = 0
    print(list)
    for i in range(len(list)):
        print(list[i])
        if(list[i] == id):
            result = 1/(i+1)
            return result
    return result

# paths


path_collection_tsv = "collection.tsv"
path_queries_train_tsv = "queries.train.tsv"
path_qid_pid_tsv = "qidpidtriples.train.full.2.tsv"
path_qrels_train_tsv = "qrels.train.tsv"
path_top1000 = "top1000.dev"

corpus = {}

with open(path_collection_tsv, 'r', encoding='utf8') as file:
    for line in file:
        pid, passage = line.strip().split("\t")
        corpus[pid] = passage

print('Loading corpus finished')

queries = {}

with open(path_queries_train_tsv, 'r', encoding='utf8') as file:
    for line in file:
        qid, query = line.strip().split("\t")
        queries[qid] = query
print('Loading queries finished')

max_training_queries = 10
max_negative_queries = 20
# Key is a query and value the amount of negative passages
negative_queries = {}
training_queries = []
counter = 0
training_data = {}
pos_pid = 0
with open(path_qid_pid_tsv, 'rt') as file:
    for line in file:
        qid, pos_id, neg_id = line.strip().split()

        if max_training_queries == counter:
            break
        if qid not in training_queries:
            qid_array = []
            qid_array.append(
                [qid, pos_id, queries[qid], corpus[pos_id], 1])
            pos_pid = pos_id
            qid_array.append(
                [qid, neg_id, queries[qid], corpus[neg_id], 0])
            counter += 1
            negative_queries[qid] = 1
            training_queries.append(qid)
            training_data[qid] = qid_array
        else:
            training_data[qid].append(
                [qid, pos_id, queries[qid], corpus[pos_id], 1])
            if(negative_queries[qid] < max_negative_queries):
                training_data[qid].append(
                    [qid, neg_id, queries[qid], corpus[neg_id], 0])
                negative_queries[qid] = negative_queries[qid] + 1

counter = 0

# with open(path_qrels_train_tsv, 'rt') as file:
#     for line in file:
#         qid, zero, pass_id, rel = line.strip().split()

#         if max_training_queries == counter:
#             break
#         training_data.append(
#             [qid, pass_id, queries[qid], corpus[pass_id], 1])
#         counter += 1
#         training_queries.append(qid)

mrr = 0

for t_query in training_queries:
    print('Loading training data finished')
    df = pd.DataFrame(training_data[t_query], columns=[
        'qid', 'pid', 'query', 'passage', 'label'])
    # preprocessing 
    stop = stopwords.words('english')
    df['passage'] = df['passage'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    df['passage'] = df['passage']. apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True)
    print(df.shape)
    print(df.head())

    # placeholder for columns
    df['LM'] = 1
    df['VSM'] = 1
    df = compute(df, 'passage', 'query')

    print(df[0:1].to_string())

    ml_data = df[['VSM', 'LM', 'Jaccard', 'label']]

    print(ml_data.head())

    # x_train, x_valid = train_test_split(
    #     ml_data, test_size=0.2, random_state=453, stratify=ml_data['label'])

    y_train = ml_data['label']
    x_train = ml_data[['VSM', 'LM', 'Jaccard']]

    loaded_model = joblib.load("filenameMNB.pkl")
    result = loaded_model.predict(x_train)
    print(result)
    probabilities = loaded_model.predict_proba(x_train)
    probabilities_df = pd.DataFrame(
        probabilities, columns=['probability_0', 'probability_1'])
    print(probabilities_df["probability_0"])
    merged = x_train.copy()
    merged["true"] = y_train
    merged["predict"] = result
    merged["prob_0"] = probabilities_df["probability_0"]
    merged["prob_1"] = probabilities_df["probability_1"]
    merged["pid"] = df["pid"]
    merged = merged.sort_values(by=["predict", "prob_1"], ascending=False)
    print(merged)
    print(average_precision_score(y_train, result))
    searched_pid = merged[merged["true"] == 1].loc[0]["pid"]
    print(searched_pid)
    calc_mrr = calculateMRR10(merged["pid"].array, searched_pid)
    print(calc_mrr)
    mrr += calc_mrr

print("===============================")
print(mrr/len(training_queries))
