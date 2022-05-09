import nltk
import time
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np

# Init stemmer
porter=PorterStemmer()
# Download once for usage of nltk
#nltk.download('punkt')

log_array = []

for i in range(0,100):
    log_array.append(np.log(i))

# Return a map with the raw term frequency for each passage
def tfPassage(passage):
        token_words = passage.split(" ")
        passage_map = {"raw-word" : {}}
        max_tf = 0
        for word in token_words:
            #stemmed_word = porter.stem(word)
            stemmed_word = word
            try:
                occurance = passage_map["raw-word"][stemmed_word] + 1
                passage_map["raw-word"][stemmed_word] = occurance
                if(max_tf < occurance):
                    max_tf = occurance
            except:
                passage_map["raw-word"][stemmed_word] = 1
                if(max_tf < 1):
                    max_tf = 1
        passage_map["max_tf"] = max_tf
        #for key, value in passage_map["raw-word"].items():
        #    passage_map["tfScore"][key] = (1+log_array[value])/(1+log_array[max_tf])

        return passage_map

# Calculate IDF value for all terms in passages
def idf(passage_tf_list):
    N = len(passage_tf_list)
    log_N = np.log(N)
    term_map = {}
    for doc in passage_tf_list:
        for key in doc["raw-word"].keys():
            try:
                term_map[key] += 1
            except:
                term_map[key] = 1
    for key in term_map:
        term_map[key] = log_N / term_map[key]
    return term_map

# Calculate the tfidf value based on the terms from the query and the precomputed tf and idf values
def tfidf(query, passage_tf_list, idf_map):
    query_words = query.split(" ")
    vector = []

    for i in range(len(query_words)):
        vector.append([])
        idf = 0
        try:
            idf = idf_map[query_words[i]]
        except:
            idf = 0
        for z in range(len(passage_tf_list)):
            try:
                vector[i].append(idf*(1+log_array[passage_tf_list[z]["raw-word"][query_words[i]]])/(1+log_array[passage_tf_list[z]["max_tf"]]))
            except:
                vector[i].append(0)
    return vector

# Calculate the cosineSimilarity for two vectord
def cosineSimilarity(document_vector, query_vector):
    result = []
    for i in range(len(document_vector[0])):
        dot_product = 0
        x_sum = 0
        y_sum = 0
        for z in range(len(query_vector)):
            doc_value = document_vector[z][i]
            dot_product += doc_value*1
            x_sum += doc_value*doc_value
            y_sum += 1
        result.append(dot_product/(np.sqrt(x_sum)*np.sqrt(y_sum)))
    return result
        

passage_tf_list = []

# Read file with all passages in order to calculate tf values
passageFile = open("collection.tsv", encoding="utf-8")
count =0
start = int(round(time.time() * 1000))
while True and count < 100000:
    # Get next line from file
    count+=1
    line = passageFile.readline()
    # if line is empty
    # end of file is reached
    if not line:
        break
    passage_tf_list.append(tfPassage(line.strip()))
 
passageFile.close()

#print(passage_tf_list)
idf_map = idf(passage_tf_list)
tfidf_vector = tfidf("presence of communication", passage_tf_list, idf_map)
cosine_similarity = cosineSimilarity(tfidf_vector, "presence of communication".split(" "))
end = int(round(time.time() * 1000))
print(end-start)
#print(tfidf_vector)
#print(idf_map)
#print(passage_tf_list)
#print(cosine_similarity)