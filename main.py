from models import *
from saveAndFetch import *
from preprocess import *
from visualize import *


#def trainModel():
def plotResults():
    makeBarChart()

def predict(m):
    input1 = input("Type your (yes/no) question: ")   
    input_vector = preprocessInput(input1)
    #print(input_vector)
    model = fetchModel(m)
    prediction = model.predict(input_vector)
    if prediction < 0.5:
        answer = "No."
    else: answer = "Yes."
    print("Answer: ", answer)


def run():
    #randomPredictor("onlyWords_Data")
    #check()
    #check("QandA_dataset")
    #loadData()
    #preprocessQuestionsAndLabels("QandA_dataset")
    #preprocessPassages("QandA_dataset")
    #preprocessQuestions("QandA_dataset")
    #findFeatures()
    #trainModel("trainingAndTesting_Data")
    #check = fetchData("QandA_dataset")
    feedforward("words")
    #plotResults()
    #lstm("words")
    #mlpClassifier("words")
   # print(check[0][1][0])
    #predict("Feedforward")
 
    
run()