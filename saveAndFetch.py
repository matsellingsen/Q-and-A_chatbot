import os
import pickle

# Fetches trained model
"""
Args:
    filename (String): Name of pickle.file containing the trained model.

Returns:
    data (object): trained model
"""
def fetchModel(filename): 
    # Get path to current working directory
    cwd = os.getcwd()

    filepath = cwd + '\\models\\' + filename + '.pickle'
    with open(filepath, 'rb') as handle:
        data = pickle.load(handle) 
    
    return data


# stores/saves trained model
"""
Args:
    filename (String): Name of pickle.file to save trained model as.
    data (object): Trained model.
"""
def storeModel(filename, data):
    # Get path to current working directory
    cwd = os.getcwd()

    filepath = cwd + '\\models\\' + filename + '.pickle'
    with open(filepath, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Fetches unprocessed data
"""
Args:
    filename (String): Name of pickle.file containing the data.

Returns:
    data (list): Data
"""
def fetchData(filename): 
    # Get path to current working directory
    cwd = os.getcwd()

    filepath = cwd + '\\data\\' + filename + '.pickle'
    with open(filepath, 'rb') as handle:
        data = pickle.load(handle) 
    
    return data


# stores/saves unprocessed data
"""
Args:
    filename (String): Name of pickle.file to save data as.
    data (list): Data
"""
def storeData(filename, data):
    # Get path to current working directory
    cwd = os.getcwd()

    filepath = cwd + '\\data\\' + filename + '.pickle'
    with open(filepath, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Fetches processed data
"""
Args:
    filename (string): Name of pickle.file containing the processed data.

Returns:
    data (list): processed data.
"""
def fetchTrainingData(filename):
    # Get path to current working directory
    cwd = os.getcwd()

    filepath = cwd + '\\data\\ready_for_training\\' + filename + '.pickle'
    with open(filepath, 'rb') as handle:
        data = pickle.load(handle) 
    
    return data


# Stores/saves processed data
"""
Args:
    filename (String): Name of pickle.file to save processed data as.
    data (list): Processed data.
"""
def storeTrainingData(filename, data):
    # Get path to current working directory
    cwd = os.getcwd()

    filepath = cwd + '\\data\\ready_for_training\\' + filename + '.pickle'
    with open(filepath, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Fetches metric-score-results
"""
Returns:
    data (list): metric-scores to trained models.
"""
def fetchResults():
    # Get path to current working directory
    cwd = os.getcwd()

    filepath = cwd + '\\data\\results\\' + 'results' + '.pickle'
    with open(filepath, 'rb') as handle:
        data = pickle.load(handle) 
    
    return data 


# Stores/saves metric-score-results
"""
Args:
    value (list): metric-score-results.
"""
def storeResults(value):

  # Fetching previous results and appending new results.
    results = fetchResults()
    iteration = len(results.keys())
    results[str(iteration)] = value
 
  # Get path to current working directory
    cwd = os.getcwd()

  # Storing updated results-file.
    filepath = cwd + '\\data\\results\\' + 'results' + '.pickle'
    with open(filepath, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Initializing results-file
"""
Run this file to reset the results-file, i.e delete all stored metric-score-results.
"""
def initResultsFile():
    # Get path to current working directory
    cwd = os.getcwd()

    results = {}
    #Storing results-file.
    filepath = cwd + '\\data\\results\\' + 'results' + '.pickle'
    with open(filepath, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

#initResultsFile()