import numpy as np
import matplotlib.pyplot as plt
from saveAndFetch import fetchResults

#Plots metric-scores for results saved in results.pickle
def makeBarChart(): 

      # Fetching saved metric-scores
        results = fetchResults()
        features = list(results.keys())
        val = list(results.values())
        accuracies = [r[1] for r in val]
        precisions = [r[2] for r in val]
        recalls = [r[3] for r in val]
        f1_scores = [r[4][0] for r in val]

      # Definining Size of plot and width to bars.
        fig = plt.figure(figsize = (10, 5))
        bar_width = 0.1

      # creating the bar plots, one for each metric-type. 
        plt.xticks(range(len(features)), features, rotation=30)
        f1_score = plt.bar(np.arange(len(features)) + bar_width*3, f1_scores, bar_width, color ='red', label="F1-score")
        recall = plt.bar(np.arange(len(features)) + bar_width*2, recalls, bar_width, color ='green', label="Recall")
        precision = plt.bar(np.arange(len(features)) + bar_width, precisions, bar_width, color ='yellow', label="Precision")
        accuracy = plt.bar(range(len(features)), accuracies, bar_width, color ='blue', label="accuracy")

      # Adding metric-values on top of their respective bars.        
        for rect in accuracy + precision + recall + f1_score:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width() / 2.0, height, str(round(height, 4)), ha='center', va='bottom')

      # Labeling x-and-y-axis, setting title, creating are that explaing each bar-type, and showing bar-chart.
        plt.xlabel("Iteration")
        plt.ylabel("Percentage")
        plt.title("Model: Feedforward")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

