import streamlit as st
import sys
from ast import operator
import os
import re
import string
import nltk.stem as ns
from nltk.stem import WordNetLemmatizer
import nltk
import math
import json
import ast # this module helps to find out programmatically what the current grammar looks like.
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('omw-1.4')

Iindex = {}
Pindex = {}
data2=[]
qvec={}
lemmatizer = WordNetLemmatizer()
dataDict = {}
stoplist = []
folderpath = r'D:/19k1279 CS-A_Assignment_1_/A1/Abstracts/'
docs = []
alpha=0.001
flag = 0

# regex and nltk used to remove punctuations

@st.experimental_memo
def remove_punctuations(contents):
    res = re.sub(r',', ' ', contents) #r means the string will be treated as raw string.
    res = re.sub(r'-', ' ', res)
    res = re.sub(r'[^\w\s]', '', res)
    translator = str.maketrans("", "", "'!@#$%^&*()_=-\|][:';:,<.>/?`~")
    x = res.translate(translator)
    return nltk.word_tokenize(x)

@st.experimental_memo
def case_fold(contents):
    contents = contents.casefold()
    return contents

@st.experimental_memo
def remove_stoplist_word(tokens):
    stoplist = open('Stopword-List.txt').read().split()
    tokens = [token for token in tokens if token not in stoplist]
    return tokens

#pre-defined porter stemmer used for stemming tokens
@st.experimental_memo
def lemmatize_tokens(tokens):
    stemmed_words = [lemmatizer.lemmatize(token) for token in tokens]
    return stemmed_words

# this function calls routines/functions to tokenize,remove stop words,stemm tokens and add them to ditcionary
@st.experimental_memo
def preprocessor(contents):
    for content in contents:
        #content[1] string ,content[0] docid
        tokens = remove_punctuations(content[1])
        filtered_tokens = remove_stoplist_word(tokens)
        stemmed_tokens = lemmatize_tokens(filtered_tokens)
        filtered_tokens1 = remove_stoplist_word(stemmed_tokens)
        dataDict[content[0]] = filtered_tokens1
    return dataDict

@st.experimental_memo
def get_vocabulary(data):
    tokens = []
    for token_list in data.values():
        tokens = tokens + token_list
    # This function is used to find the frequency of words within a text. It returns a dictionary. We need to pass keys and values to get the data.
    fdist = nltk.FreqDist(tokens)
    return list(fdist.keys())

@st.experimental_memo
def inverted_index(data2, data):
    for word in data2:
        for dId, tokens in data.items():
            if word in tokens:
                if word in Iindex.keys():
                    if Iindex[word]['tf'][dId] == 0 : #in new doc
                        Iindex[word]['df'] = Iindex[word]['df'] + 1
                        Iindex[word]['tf'][dId] = 1
                    else :
                        Iindex[word]['tf'][dId] = Iindex[word]['tf'][dId] +1 
                else:
                    Iindex[word]={
                        'tf' : [0]*449,  #term frequency
                        'df' : 0,        #document frequency
                        'idf':0,         #inverse document frequency
                        'tf-idf':[0]*449  # tf-idf weights
                    }
                    Iindex[word]['tf'][dId] = 1
                    Iindex[word]['df'] = 1
    for dId in range(448):
        for word in Iindex:
            Iindex[word]['idf'] = math.log(448/(Iindex[word]['df']) , 10 )     #log(n/df)
            Iindex[word]['tf-idf'][dId +1 ] = Iindex[word]['tf'][dId+1] * Iindex[word]['idf']  #tf*idf scoring
            
    return Iindex


def filereader():
    global Iindex
    global Pindex
    words = []
    count = 1
    file_exists = os.path.exists("inverted_index.txt")
    if file_exists:  # if file exists load inverted index from file
        #print('hellllll')
        with open('inverted_index.txt') as f:
            data = f.read()
            Iindex = ast.literal_eval(data) #ast. literal_eval raises an exception if the input isn't a valid Python datatype, so the code won't be executed if it's not
    else:  # else create and save index to file
        for f in os.listdir(folderpath):
            data = case_fold(open('D:\\SEM 6\\IR\\A2\\Abstracts\\Abstracts\\' +
                                f, 'r', encoding='utf-8', errors='ignore').read())
            docs.append(int(f[:-4]))#extract dicId from filename 
            words.append((int(f[:-4]), data))
            count += 1
        data = preprocessor(words)
        data2 = get_vocabulary(data)
        inverted_index(data2, data)
        with open('inverted_index.txt', 'w') as f:
            f.write(json.dumps(Iindex)) # write to file in json format,easy to read from file afterwards
        f.close()
    file_exists = os.path.exists("positional_index.txt")
    
    return words


# handle general queries
@st.experimental_memo
def processQeury(query):
    queryterms = []
    lemmatized_Qterms = []
    x = case_fold(query)
    queryterms = remove_punctuations(x)
    lemmatized_Qterms = lemmatize_tokens(queryterms)
    
    print(lemmatized_Qterms)
    for word in Iindex.keys():
        qvec[word] = 0
            
    for word in lemmatized_Qterms:
        if word in qvec:
            qvec[word]=(lemmatized_Qterms.count(word)*Iindex[word]['idf'])
            
    dotprod = 0.0
    vec1 = 0.0
    vec2 = 0.0
    res=[]
    for dId in range(448):
        for word in qvec:
            if word not in stoplist:
                dotprod += (qvec[word] * Iindex[word]['tf-idf'][dId+1])  #index is doc is word tf-idf
                vec1 += qvec[word]**2
                vec2 += Iindex[word]['tf-idf'][dId+1]**2
        res.append(dotprod / (math.sqrt(vec1) * math.sqrt(vec2) ) )
        dotprod = 0.0
        vec1 = 0.0
        vec2 = 0.0
    print(res)
    doc = []
    docvals=[]
    f=1
    for i in res:
        if i > alpha:
            docvals.append(i)
            doc.append(f)
        f = f +1    
    return doc,docvals


# this function validates the existence of files containing inverted index and positional index,if they exist it reads and loads the indexes from file,else creates both indexes and saves them in separate files

filereader()

# GUI



    