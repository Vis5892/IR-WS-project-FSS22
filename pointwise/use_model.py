import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from compute_LM_VSM_Jaccard import compute
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

# function for calculating MRR10 score

def calculateMRR10(list, id):
    result = 0
    #print(list)
    for i in range(len(list)):
        #print(list[i])
        if(list[i] == id):
            result = 1/(i+1)
            return result
    return result



# paths of the data files used

path_collection_tsv = "collection.tsv"
path_queries_train_tsv = "queries.train.tsv"
path_qid_pid_tsv = "qidpidtriples.train.full.2.tsv"
path_qrels_train_tsv = "qrels.train.tsv"
path_top1000 = "top1000.dev"


# Load corpus for test data

corpus = {}

with open(path_collection_tsv, 'r', encoding='utf8') as file:
    for line in file:
        pid, passage = line.strip().split("\t")
        corpus[pid] = passage

print('Loading corpus finished')

# Load queries

queries = {}

with open(path_queries_train_tsv, 'r', encoding='utf8') as file:
    for line in file:
        qid, query = line.strip().split("\t")
        queries[qid] = query
print('Loading queries finished')

# Load test data out of qid_pid

max_test_queries = 10          # test queries used
max_negative_queries = 20        # amount of negative passages related to query for test data
# Key is a query and value the amount of negative passages
negative_queries = {}
test_queries = []
counter = 0
test_data = {}
pos_pid = 0
with open(path_qid_pid_tsv, 'rt') as file:
    for line in file:
        qid, pos_id, neg_id = line.strip().split()

        if max_test_queries == counter:
            break
        if qid not in test_queries:
            qid_array = []
            qid_array.append(
                [qid, pos_id, queries[qid], corpus[pos_id], 1])
            pos_pid = pos_id
            qid_array.append(
                [qid, neg_id, queries[qid], corpus[neg_id], 0])
            counter += 1
            negative_queries[qid] = 1
            test_queries.append(qid)
            test_data[qid] = qid_array
        else:
            test_data[qid].append(
                [qid, pos_id, queries[qid], corpus[pos_id], 1])
            if(negative_queries[qid] < max_negative_queries):
                test_data[qid].append(
                    [qid, neg_id, queries[qid], corpus[neg_id], 0])
                negative_queries[qid] = negative_queries[qid] + 1

print('Loading test data finished')
counter = 0
mrr = 0

for t_query in test_queries:
    df = pd.DataFrame(test_data[t_query], columns=[
        'qid', 'pid', 'query', 'passage', 'label'])
    # preprocessing
    stop = stopwords.words('english')
    df['passage'] = df['passage'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    df['passage'] = df['passage']. apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True)
    #print(df.shape)
    #print(df.head())

    # placeholder for columns
    df['LM'] = 1
    df['VSM'] = 1
    df['Jaccard'] = 1
    df = compute(df, 'passage', 'query')

    #print(df[0:1].to_string())

    # create machine learning dataframe

    ml_data = df[['VSM', 'LM', 'Jaccard', 'label']]

    #print(ml_data.head())

    # create x_test and y_test
    y_test = ml_data['label']
    x_test = ml_data[['VSM', 'LM', 'Jaccard']]

    # load best model
    loaded_model = joblib.load("filenameMNB.pkl")

    # predict labels
    result = loaded_model.predict(x_test)
    #print(result)

    # predict probabilities
    probabilities = loaded_model.predict_proba(x_test)
    probabilities_df = pd.DataFrame(
        probabilities, columns=['probability_0', 'probability_1'])
    #print(probabilities_df["probability_0"])
    merged = x_test.copy()
    merged["true"] = y_test
    merged["predict"] = result
    merged["prob_0"] = probabilities_df["probability_0"]
    merged["prob_1"] = probabilities_df["probability_1"]
    merged["pid"] = df["pid"]
    merged = merged.sort_values(by=["predict", "prob_1"], ascending=False)

    #print(average_precision_score(y_test, result))
    searched_pid = merged[merged["true"] == 1].loc[0]["pid"]

    #print(searched_pid)

    calc_mrr = calculateMRR10(merged["pid"].array, searched_pid)
    #print(calc_mrr)
    mrr += calc_mrr

# print final MRR10
print("MRR10:")
print(mrr/len(test_queries))