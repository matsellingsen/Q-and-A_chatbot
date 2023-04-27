
import os
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
import kerastuner as kt
from tensorflow import keras
from saveAndFetch import *
import numpy as np
import random as rn

# Classifies questions randomly as 1 or 0.
"""
Args:
    filename (String): name of pickle-file the decired preprocessed data.

The point of this file was to validate that a random predictor in fact outputs and accuracy of  approx. 50%.
"""
def randomPredictor(filename):
    data = fetchData(filename)
    Y_train, Y_test = data[0], data[1], data[2], data[3]

    Y = Y_train + Y_test

    riktig = 0
    for y in Y:
        randomNumber = rn.randint(0,1)
        if randomNumber == y:
            riktig += 1
    
    print("Accuracy for random predictor: ", riktig/len(Y))


# Trains MLPCLassifier, and stores the trained model as a pickle-file.
"""
Args:
    filename (String): name of pickle-file the decired preprocessed data.
"""
def mlpClassifier(filename):
  # Fetching both training-and-test-data
    data = fetchTrainingData(filename)
    X_train, X_test, Y_train, Y_test = data[0], data[1], data[2], data[3]

  # Initializing model
    model = MLPClassifier(activation="relu", solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(16, 3), random_state=42)

  # training model
    trainedModel = model.fit(X_train, Y_train)

  # calculating the models metric-scores based on its performance on the test-set. 
    predicted = trainedModel.predict(X_test)
    correct = 0
    for i in range(len(predicted)):
        if predicted[i] == Y_test[i]:
            correct += 1
    print(f"Accuracy: ", correct/len(predicted))
    print(f"Precision: ", str(precision_score(Y_test, predicted, average=None, zero_division=0)[1])) 
    print(f"Recall: ", str(recall_score(Y_test, predicted, average=None, zero_division=0)[1])) 
    print(f"f1-score: ", str(f1_score(Y_test, predicted, average=None)[1]))
    
  # Fetching metric-values
    accuracy = correct/len(predicted)
    precision = precision_score(Y_test, predicted, average=None)
    recall = recall_score(Y_test, predicted, average=None)
    f1Score = f1_score(Y_test, predicted, average=None)
    metrics = [accuracy, precision, recall, f1Score]

  # Saving trained model
    storeModel("MLPClassifier", model)

  # Saving metric-score-results
    storeResults(metrics)


# Trains Feedforward-model, and stores the trained model as a pickle-file.
"""
Args:
    filename (String): name of pickle-file the decired preprocessed data.
"""
def feedforward(filename):
  # Fetching both training-and-test-data
    data = fetchTrainingData(filename)
    X_train, X_test, Y_train, Y_test = np.array(data[0]), np.array(data[1]), np.array(data[2]), np.array(data[3])

# Converting labels from int to float, so they are compatible with Keras' fit()-and-evaluate()-functions.
    Y_train = Y_train.astype(np.float32)
    Y_test = Y_test.astype(np.float32)

  # Finding input shape for model
    inputShape = len(X_train[0])

  # Setting fixed random seed to get reproducible results.
    os.environ['PYTHONHASHSEED']=str(42)
    np.random.seed(42)
    rn.seed(42)
    tf.random.set_seed(42)

  # Building model
    model = keras.Sequential()
    model.add(keras.layers.Dense(500, activation="relu", input_shape=(inputShape,)))   
    model.add(keras.layers.Dense(1000, activation="relu"))
    model.add(keras.layers.Dense(1000, activation="relu"))
    model.add(keras.layers.Dense(1000, activation="relu"))
    model.add(keras.layers.Dense(1000, activation="relu"))
    model.add(keras.layers.Dense(1000, activation="relu"))
    model.add(keras.layers.Dense(1000, activation="relu"))
    model.add(keras.layers.Dense(1000, activation="relu"))
    model.add(keras.layers.Dense(1000, activation="relu"))
    model.add(keras.layers.Dense(1000, activation="relu"))
    model.add(keras.layers.Dense(1000, activation="relu"))
    model.add(keras.layers.Dense(1000, activation="relu"))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

  # Compiling model and defining optimizer, learning-rate, loss-function and metrics.
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(), keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.F1Score(threshold=0.5)])

  # fetching class weights
    weights = calculateClassWeights(filename)
    classWeights = {0: weights[0], 1: weights[1]}

  # Training model
    model.fit(x=X_train, y=Y_train, epochs=1, batch_size=64, class_weight=classWeights)

  # Predicting on test-set.
    predictions = model.predict(X_test)
    counterZero = 0
    counterOne = 0
    for pred in predictions:
        if pred < 0.5:
            counterZero += 1
        else:
            counterOne += 1

  # Uncomment these to print how many "0" and "1" the model predicted.
    #print("zero_predictions: ", counterZero)
    #print("one_predictions: ", counterOne)

  # Evaluating model
    metrics = model.evaluate(X_test, Y_test)
    print("Accuracy: ", metrics[1])
    print("Precision: ", metrics[2])
    print("Recall: ", metrics[3])
    print("F1-score: ", metrics[4][0])

    #Saving trained model
    storeModel("Feedforward", model)

    #Saving results
    storeResults(metrics)


# Trains LSTM, and stores the trained model as a pickle-file.
"""
Args:
    filename (String): name of pickle-file the decired preprocessed data.
"""
def lstm(filename):

  # Fetching both training-and-test-data
    data = fetchTrainingData(filename)
    X_train, X_test, Y_train, Y_test = np.array(data[0]), np.array(data[1]), np.array(data[2]), np.array(data[3])

  # Converting labels from int to float, so they are compatible with Keras' fit()-and-evaluate()-functions.
    Y_train = Y_train.astype(np.float32)
    Y_test = Y_test.astype(np.float32)

  # Finding input shape for model
    inputShape = len(X_train[0])

  # Setting fixed random seed to get reproducible results.
    os.environ['PYTHONHASHSEED']=str(42)
    np.random.seed(42)
    rn.seed(42)
    tf.random.set_seed(42)

  # Building model
    model = keras.Sequential()
    model.add(keras.layers.LSTM(500, activation="relu", recurrent_activation="sigmoid", input_shape=(inputShape,1)))
    model.add(keras.layers.Dense(250, activation='relu'))   
    model.add(keras.layers.Dense(250, activation='relu'))
    model.add(keras.layers.Dense(50, activation='relu'))
    model.add(keras.layers.Dense(50, activation='relu'))
    model.add(keras.layers.Dense(50, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

  # Compiling model and defining optimizer, learning-rate, loss-function and metrics.
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(), keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.F1Score(threshold=0.5)])

  # fetching class weights
    weights = calculateClassWeights(filename)
    classWeights = {0: weights[0], 1: weights[1]}

  # Training model
    model.fit(x=X_train, y=Y_train, epochs=1, batch_size=64, class_weight=classWeights)
    print("check3")

  # Evaluating model
    metrics = model.evaluate(X_test, Y_test)
    print("Accuracy: ", metrics[1])
    print("Precision: ", metrics[2])
    print("Recall: ", metrics[3])
    print("F1-score: ", metrics[4][0])

  # Saving trained model
    storeModel("LSTM", model)

  # Saving results
    storeResults(metrics)


# Calculates appropriate class-weights for each class.
"""
Args:
    filename (String): name of pickle-file the decired preprocessed data.

Returns:
    (tuple): Calculated classweights.
"""
def calculateClassWeights(filename):
    data = fetchTrainingData(filename)
    Y_train = data[2]
    zeros_train = ones_train = 0
    for elem in Y_train:
        if elem == 0:
            zeros_train += 1
        else:
            ones_train += 1

    zero_weight = (1 / zeros_train) * (len(Y_train)/2)  
    one_weight = (1 / ones_train) * (len(Y_train)/2)

# Uncomment these to print the calculated class-weights.
    #print(zero_weight)
    #print(one_weight)

    return (zero_weight, one_weight)

