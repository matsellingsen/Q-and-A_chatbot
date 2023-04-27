from models import *
from saveAndFetch import *
from preprocess import *
from visualize import *

""" 
This projects root-file.

All functions are called and run from this file.
       
"""

# Plots metric-scores for trained models with. See visualize.py for further info.
def plotResults():
    makeBarChart()


# Predicts answers based on user-input.
""" 
Args:
    m (String) : "Feedforward", "LSTM" or "MLPClassifier"

Return:
    Prints answer to console.

"""
def predict(m):
    input1 = input("Type your (yes/no) question: ")   
    input_vector = preprocessInput(input1)
    model = fetchModel(m)
    prediction = model.predict(input_vector)
    if prediction < 0.5:
        answer = "No."
    else: answer = "Yes."
    print("Answer: ", answer)



#Main function for running preprocessing, training, plotting and prediction.
def run():

    """ Uncomment to load the Boolq-datset: 
        NB! The dataset should already be loaded and saved as a pickle-file, so this function is typically
            not necessary to run.
    """
    #loadData()
    

    """ Uncomment to preprocess : 
        NB! All iterations of processed data which is mentioned in the report should already be performed
            and saved as pickle-files in the ready_for_training-folder. 
            The names of these pickle-files indicate what features they have. 
            E.g. "words_length.pickle" contain words and question-length-count as features.
        Args:
            See preprocess.py
    """
    #preprocessQuestionsAndLabels("QandA_dataset", "2")


    """ Uncomment the model you want to train: 
        Args:
            See models.py
    """
    #randomPredictor("onlyWords_Data")
    #mlpClassifier("words")
    #feedforward("length_stopwords_ner_pos_neg")
    #lstm("words")
   

    """
        Uncomment to plot metric-scores of trained models. 
    """
    #plotResults()


    """
        Uncomment to predict answer from user-input. 
        Args:
            See predict-function in this file.
    """
    #predict("Feedforward")
     
run()