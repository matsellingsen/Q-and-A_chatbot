import os
import pickle
import re
from datasets import load_dataset
import nltk
from nltk import sent_tokenize as st, word_tokenize as wt
from nltk.corpus import stopwords as sw
from string import punctuation
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import spacy as sp
from negspacy.negation import Negex
from transformers import AutoTokenizer, AutoModelForMaskedLM
from saveAndFetch import *
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.decomposition import TruncatedSVD


def loadData():
    # Get path to current working directory
    cwd = os.getcwd()

    #Load dataset
    dataset = load_dataset("super_glue", "boolq")

    #Split questions, passages and labels into different lists
    questions = [elem for elem in dataset["train"]["question"]]
    questions += [elem for elem in dataset["validation"]["question"]]

    passages = [elem for elem in dataset["train"]["passage"]]
    passages += [elem for elem in dataset["validation"]["passage"]]

    labels = [elem for elem in dataset["train"]["label"]]
    labels += [elem for elem in dataset["validation"]["label"]]
    
    #Storing data in a local Pickle-file
    loadedData = [questions, passages, labels]
    with open(cwd + '\\data\\QandA_dataset.pickle', 'wb') as handle:
        pickle.dump(loadedData, handle, protocol=pickle.HIGHEST_PROTOCOL)
   # return questions, passages, labels


def extractFeatures(question, spacyPipeline):
    features = {}
    ner_tags = spacyPipeline.get_pipe('ner').labels
    pos_tags = spacyPipeline.get_pipe('tagger').labels
    features.update({nertag: 0 for nertag in ner_tags})
    features.update({postag: 0 for postag in pos_tags})

    #Sentence length
    questionLength = len(question)
    features["questionLength"] = questionLength

    #stopword count
    stopwords = len([token for token in question if token in sw.words("english")])
    features["stopwords"] = stopwords

    #Running through spacy-pipeline
    questionSpacy = spacyPipeline(" ".join(question))
    negations = 0
    for token in questionSpacy:
        #NER-tags
        if token.ent_type_:
            features[token.ent_type_] += 1
        
        #POS-tags
        features[token.tag_] += 1

        if token.dep_ == "neg":
          #  print("negation: ", token)
          #  print("NEGATION: ", token.text)
            negations += 1
        features["negations"] = negations

    features_list = list(features.items())

    return features_list

def preprocessInput(input):
    spacyPipeline = sp.load("en_core_web_sm", disable=["tokenizer"])
    input_features = extractFeatures(wt(input), spacyPipeline)
    input_to_vec = [[feature[1] for feature in input_features]]

    return input_to_vec

def preprocessQuestionsAndLabels(filename):
   
    data = fetchData(filename)
    Q, L = data[0], data[2]

    print("Q: ", len(Q))
    print("L: ", len(L))

    i = 0
    processedQ = []
   
    spacyPipeline = sp.load("en_core_web_sm", disable=["tokenizer"])
    i = 0
    for question in Q:
        print(i)
        tokenized_q = wt(question)
        q_features = extractFeatures(tokenized_q, spacyPipeline)
        processedQ.append([feature[1] for feature in q_features])
        i += 1

    for Q in processedQ[:10]:
        print(len(Q))
        print(Q) 
  
    # split train and test data
    X_train, X_test, y_train, y_test = train_test_split(processedQ, L, test_size=0.2, random_state=42)

    # Store in pickle-file
    trainingAndTestingData = [X_train, X_test, y_train, y_test]
    storeTrainingData("length_ner_pos_neg", trainingAndTestingData)
    
    return Q


