import os
import pickle


def fetchModel(filename): 
    # Get path to current working directory
    cwd = os.getcwd()

    filepath = cwd + '\\models\\' + filename + '.pickle'
    with open(filepath, 'rb') as handle:
        data = pickle.load(handle) 
    
    return data

def storeModel(filename, data):
    # Get path to current working directory
    cwd = os.getcwd()

    filepath = cwd + '\\models\\' + filename + '.pickle'
    with open(filepath, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def fetchData(filename): 
    # Get path to current working directory
    cwd = os.getcwd()

    filepath = cwd + '\\data\\' + filename + '.pickle'
    with open(filepath, 'rb') as handle:
        data = pickle.load(handle) 
    
    return data

def storeData(filename, data):
    # Get path to current working directory
    cwd = os.getcwd()

    filepath = cwd + '\\data\\' + filename + '.pickle'
    with open(filepath, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def fetchTrainingData(filename):
    # Get path to current working directory
    cwd = os.getcwd()

    filepath = cwd + '\\data\\ready_for_training\\' + filename + '.pickle'
    with open(filepath, 'rb') as handle:
        data = pickle.load(handle) 
    
    return data

def storeTrainingData(filename, data):
    # Get path to current working directory
    cwd = os.getcwd()

    filepath = cwd + '\\data\\ready_for_training\\' + filename + '.pickle'
    with open(filepath, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def fetchResults():
    # Get path to current working directory
    cwd = os.getcwd()

    filepath = cwd + '\\data\\results\\' + 'results' + '.pickle'
    with open(filepath, 'rb') as handle:
        data = pickle.load(handle) 
    
    return data 

def storeResults(value):

    #Fetching previous results and appending new results.
    results = fetchResults()
    iteration = len(results.keys())
    results[str(iteration)] = value
 

    # Get path to current working directory
    cwd = os.getcwd()

    #Storing updated results-file.
    filepath = cwd + '\\data\\results\\' + 'results' + '.pickle'
    with open(filepath, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

def initResultsFile():
    # Get path to current working directory
    cwd = os.getcwd()

    results = {}
    #Storing results-file.
    filepath = cwd + '\\data\\results\\' + 'results' + '.pickle'
    with open(filepath, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

#initResultsFile()