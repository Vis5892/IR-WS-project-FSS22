import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from compute_LM_VSM import compute
from sklearn.metrics import average_precision_score
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
import tarfile
import gzip


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

max_training_queries = 1000
max_negative_queries = 5
# Key is a query and value the amount of negative passages
negative_queries = {}
training_queries = []
counter = 0
training_data = []

with open(path_qid_pid_tsv, 'rt') as file:
    for line in file:
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

counter = 0

with open(path_qrels_train_tsv, 'rt') as file:
    for line in file:
        qid, zero, pass_id, rel = line.strip().split()

        if max_training_queries == counter:
            break
        training_data.append(
            [qid, pass_id, queries[qid], corpus[pass_id], 1])
        counter += 1
        training_queries.append(qid)

print('Loading training data finished')
df = pd.DataFrame(training_data, columns=[
                  'qid', 'pid', 'query', 'passage', 'label'])
df.drop_duplicates(inplace=True)
df.reset_index(inplace=True)
print(df.shape)
print(df.head())

# placeholder for columns
df['LM'] = 1
df['VSM'] = 1
df = compute(df, 'passage', 'query')

print(df[0:1].to_string())

ml_data = df[['VSM', 'LM', 'label']]

print(ml_data.head())

x_train, x_valid = train_test_split(
    ml_data, test_size=0.2, random_state=453, stratify=ml_data['label'])

y_train = x_train['label']
x_train = x_train[['VSM', 'LM']]

y_valid = x_valid['label']
x_valid = x_valid[['VSM', 'LM']]

mnb = MultinomialNB()
mnb.fit(x_train, y_train)

# Predict labels and print AP
predict = mnb.predict(x_valid)
merged = x_valid.copy()
merged["predict"] = predict
print("Predictions")
print(merged.loc[merged["predict"] == 1])
print(average_precision_score(y_valid, predict))

# svc = svm.SVC(probability=True)
# svc.fit(x_train, y_train)

# dtc = DecisionTreeClassifier(random_state=0)
# dtc.fit(x_train, y_train)

re_ranking_data = []

with open(path_top1000, 'r', encoding='utf8') as file:
    for line in file:
        qid, pid, query, passage = line.strip().split("\t")
        re_ranking_data.append([qid, pid, query, passage])

print('Loading reranking data finished')
df_re_rank = pd.DataFrame(re_ranking_data, columns=[
    'qid', 'pid', 'query', 'passage'])
print(df_re_rank.shape)
print(df_re_rank.head())

# placeholder for columns
df_re_rank = df_re_rank.loc[0:10000]
df_re_rank['LM'] = 1
df_re_rank['VSM'] = 1
df_re_rank = compute(df_re_rank, 'passage', 'query')
print('converting reranking data')
# convert to ml data for ranking
ml_re_rank = df_re_rank[['VSM', 'LM']]

probabilities = mnb.predict_proba(ml_re_rank)
probabilities_df = pd.DataFrame(
    probabilities, columns=['probability_class1', 'probability_class2'])
ml_re_rank['prediction_proba'] = probabilities_df['probability_class1']

# probabilities_svc = svc.predict_proba(df_re_rank[['VSM', 'LM']])
# probabilities_svc_df = pd.DataFrame(
#     probabilities_svc, columns=['probability_class1', 'probability_class2'])
# ml_re_rank['prediction_svc_proba'] = probabilities_svc_df['probability_class1']

# probabilities_dtc = dtc.predict_proba(df_re_rank[['VSM', 'LM']])
# probabilities_dtc_df = pd.DataFrame(
#     probabilities_dtc, columns=['probability_class1', 'probability_class2'])
# ml_re_rank['prediction_dtc_proba'] = probabilities_dtc_df['probability_class1']

print(probabilities)
print(df.head())
print(probabilities_df.head())
print(ml_re_rank.head())
