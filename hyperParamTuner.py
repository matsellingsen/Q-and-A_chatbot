
import os
import numpy as np
from models import calculateClassWeights
import tensorflow as tf
import kerastuner as kt
from tensorflow import keras
import random as rn
from saveAndFetch import fetchTrainingData, storeModel, fetchData


def feedforward(hp):

    #Setting fixed random seed to get reproducible results.
    os.environ['PYTHONHASHSEED']=str(42)
    np.random.seed(42)
    rn.seed(42)
    tf.random.set_seed(42)
    layer_dimensionalities = [20, 50, 100, 200, 300, 400, 500]
    #Initializing Keras Tuner Object and hyperparameters to be tuned 
    units1 = hp.Choice("units_layer1", layer_dimensionalities)
    units2 = hp.Choice("units_layer2", layer_dimensionalities)
    units3 = hp.Choice("units_layer3", layer_dimensionalities)
    units4 = hp.Choice("units_layer4", layer_dimensionalities)
    units5 = hp.Choice("units_layer5", layer_dimensionalities)
    units6 = hp.Choice("units_layer6", layer_dimensionalities)
    #learning_rate = hp.Float("learning_rate", min_value=1e-5, max_value=1e-2, sampling="log")
    learning_rate = hp.Choice("learning_rate", [1e-5, 1e-4, 1e-3, 1e-2])
    activation_function = hp.Choice("activation_function", ["relu", "tanh"])
    
    #Building model
    model = keras.Sequential()
    model.add(keras.layers.Dense(units1, activation=activation_function, input_shape=(1000,)))   
    model.add(keras.layers.Dense(units2, activation=activation_function))
    model.add(keras.layers.Dense(units3, activation=activation_function))
    model.add(keras.layers.Dense(units4, activation=activation_function))
    model.add(keras.layers.Dense(units5, activation=activation_function))
    model.add(keras.layers.Dense(units6, activation=activation_function))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

  
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(), keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.F1Score()])

    #Returning model
    return model

def run():
    #Fetching data
    data = fetchTrainingData("words")
    X_train, X_test, Y_train, Y_test = np.array(data[0]), np.array(data[1]), np.array(data[2]), np.array(data[3])

    Y_train = Y_train.astype(np.float32)
    Y_test = Y_test.astype(np.float32)
    
    #fetching class weights
    weights = calculateClassWeights("words")
    classWeights = {0: weights[0], 1: weights[1]}

    cd = os.getcwd()

    #Initializing Keras tuner
    tuner = kt.BayesianOptimization(
    hypermodel=feedforward,
    objective="binary_accuracy",
    max_trials=100,
    executions_per_trial=1,
    overwrite=True,
    directory=cd,
    project_name="hyperParamTuner",
    )


    tuner.search(X_train, Y_train, epochs=1, class_weight=classWeights, batch_size=64, validation_data=(X_test, Y_test))
    tuner.results_summary()

run()




