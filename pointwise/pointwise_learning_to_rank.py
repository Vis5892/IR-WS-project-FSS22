from unittest import skip
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


# paths of the data files used

path_collection_tsv = "collection.tsv"
path_queries_train_tsv = "queries.train.tsv"
path_qid_pid_tsv = "qidpidtriples.train.full.2.tsv"
path_qrels_train_tsv = "qrels.train.tsv"
path_top1000 = "top1000.dev"

# Load corpus of passages

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


# Load training data out of qid_pid

max_training_queries = 1000    # training queries used
max_negative_queries = 20      # amount of negative passages related to query for training data

# Key is a query and value the amount of negative passages
negative_queries = {}
training_queries = []
counter = 0
training_data = []
skip_lines = 0
with open(path_qid_pid_tsv, 'rt') as file:
    for line in file:
        if(skip_lines > 10):
            qid, pos_id, neg_id = line.strip().split()

            if max_training_queries == counter:
                break
            if qid not in training_queries:
                training_data.append(
                    [qid, pos_id, queries[qid], corpus[pos_id], 1])
                training_data.append(
                    [qid, neg_id, queries[qid], corpus[neg_id], 0])
                counter += 1
                negative_queries[qid] = 1
                training_queries.append(qid)
            else:
                training_data.append(
                    [qid, pos_id, queries[qid], corpus[pos_id], 1])
                if(negative_queries[qid] < max_negative_queries):
                    training_data.append(
                        [qid, neg_id, queries[qid], corpus[neg_id], 0])
                    negative_queries[qid] = negative_queries[qid] + 1
        else:
            skip_lines += 1

counter = 0
print('Loading training data finished')

# create dataframe for training data
df = pd.DataFrame(training_data, columns=[
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
df['Jaccard'] = 1
df = compute(df, 'passage', 'query')

print(df[0:1].to_string())

# create machine learning dataframe for training

ml_data = df[['VSM', 'LM', 'Jaccard', 'label']]

print(ml_data.head())

# create x_train and y_train
y_train = ml_data['label']
x_train = ml_data[['VSM', 'LM', 'Jaccard']]

# undersample data
rus = RandomUnderSampler()
x_rus, y_rus = rus.fit_resample(x_train, y_train)

# use Multinomial NB
print("Multinomial NB")
mnb = MultinomialNB()
param_search = {
    'alpha': [0.0, 0.25, 0.5, 0.75, 1.0]}

# use Decision Tree
#print("TREE")
#tree = DecisionTreeClassifier(max_depth=5, max_features="auto")
# param_search = {
#     'max_features': ['auto', "sqrt", "log2"],
#     'max_depth': [5, 15, 25, 35, 45]}

# use KNN
# print("KNN")
# knn = KNeighborsClassifier()
#param_search = {}

# Grid Search  -> best model and save it
gscv = GridSearchCV(mnb, param_grid=param_search,scoring="average_precision", cv=5)
gscv.fit(x_rus, y_rus)
best_score = gscv.best_score_
best_model = gscv.best_estimator_
print(best_score)
print(best_model)
joblib.dump(best_model, 'filenameMNB.pkl')
