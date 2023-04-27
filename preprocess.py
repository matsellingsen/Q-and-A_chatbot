import os
import pickle
from datasets import load_dataset
from nltk import word_tokenize as wt
from nltk.corpus import stopwords as sw
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import spacy as sp
from transformers import AutoTokenizer
from saveAndFetch import *
from sklearn.decomposition import TruncatedSVD


# Loads Boolq-dataset
def loadData():

    # Get path to current working directory
    cwd = os.getcwd()

    #Load dataset
    dataset = load_dataset("super_glue", "boolq")

  # Split questions, passages and labels into different lists
    questions = [elem for elem in dataset["train"]["question"]]
    questions += [elem for elem in dataset["validation"]["question"]]

    passages = [elem for elem in dataset["train"]["passage"]]
    passages += [elem for elem in dataset["validation"]["passage"]]

    labels = [elem for elem in dataset["train"]["label"]]
    labels += [elem for elem in dataset["validation"]["label"]]
    
  # Storing data in a local Pickle-file
    loadedData = [questions, passages, labels]
    with open(cwd + '\\data\\QandA_dataset.pickle', 'wb') as handle:
        pickle.dump(loadedData, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Extracts features from question
"""
Args:
    question (list): List containing a tokenized question, each word is a string.
    spacyPipeline (nlp object): space-pipeline-object
"""
def extractFeatures(question, spacyPipeline):

  # Initializing dictionary used for converting list of strings to vector.  
    features = {}
    ner_tags = spacyPipeline.get_pipe('ner').labels
    pos_tags = spacyPipeline.get_pipe('tagger').labels
    features.update({nertag: 0 for nertag in ner_tags})
    features.update({postag: 0 for postag in pos_tags})

  # Sentence length
    questionLength = len(question)
    features["questionLength"] = questionLength

  # stopword count
    stopwords = len([token for token in question if token in sw.words("english")])
    features["stopwords"] = stopwords

  # Running through spacy-pipeline
    questionSpacy = spacyPipeline(" ".join(question))
    negations = 0
    for token in questionSpacy:

      # NER-tags
        if token.ent_type_:
            features[token.ent_type_] += 1
        
      # POS-tags
        features[token.tag_] += 1

      # Negation-tags
        if token.dep_ == "neg":
            negations += 1
        features["negations"] = negations

  # Dict to list
    features_list = list(features.items())

    return features_list


# Preprocessing user-input
"""
Args: 
    Input (String): input from user.

Returns:
    input_to_vec (list): processed user-input
"""
def preprocessInput(input):
    spacyPipeline = sp.load("en_core_web_sm", disable=["tokenizer"])
    input_features = extractFeatures(wt(input), spacyPipeline)
    input_to_vec = [[feature[1] for feature in input_features]]

    return input_to_vec

def preprocessQuestionsAndLabels(filename, processingPath):

  # Fetching data, 
  # Q = list of question, L = list of labels. L[0] = Q[0]'s label.  
    data = fetchData(filename)
    Q, L = data[0], data[2]

  # Initializing list for processed questions and space-pipeline
    processedQ = []
    spacyPipeline = sp.load("en_core_web_sm", disable=["tokenizer"])

  # Processing path 1 (See report)
    if processingPath == "1":
      # Initializing tokenizer and factorizer and getting tokenizers vocabulary.
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        factorizer = TruncatedSVD(n_components=1000, random_state=42)
        vocab = tokenizer.get_vocab()
       
      # Lowercasing all words in vocab, because all questions are lowercased.
        lowercase_vocabulary = {k.lower(): v for k, v in vocab.items()}

      # Initializing vectorizer.
        vectorizer = CountVectorizer(vocabulary=lowercase_vocabulary)

      # Tokenizing all questions.
        i = 0
        for question in Q:
            print(i)
            tokenized_q = tokenizer.tokenize(question)
            processedQ.append(" ".join(tokenized_q))
            i += 1

      # Converting list of tokenized questions into matrix and redusing matrix's dimensionality from 30 522 -> 1 000.
        questionMatrices = vectorizer.fit_transform(processedQ).toarray()
        processedQ = factorizer.fit_transform(questionMatrices)
        print("success")

  # Processing path 2 (see report)
    else: 
      # Tokenizing all questions and extracting features.
        for question in Q:

            tokenized_q = wt(question)
            q_features = extractFeatures(tokenized_q, spacyPipeline)
            processedQ.append([feature[1] for feature in q_features])

  # split train and test data.
    X_train, X_test, y_train, y_test = train_test_split(processedQ, L, test_size=0.2, random_state=42)

  # Store in pickle-file.
    trainingAndTestingData = [X_train, X_test, y_train, y_test]
    storeTrainingData("test", trainingAndTestingData)
    
    return Q


