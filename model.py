#TODO
# Jobb med simple-wikipedia dataset, finn ut om det kan brukes.

from lxml import objectify
import xml.etree.ElementTree as ET
import xmltodict
import pandas as pd
import os
import pickle
import spacy as sp
import re
import nltk
from nltk import sent_tokenize as st, word_tokenize as wt



def readData():
    allArticles = {}
    # Get path to current working directory
    cwd = os.getcwd()

    # Open the XML file
    tree = ET.parse(cwd + '\\data\\simplewiki_articles.xml')

    # Get the root element
    root = tree.getroot()

    # Loop through each page element in the XML file
    for page in root.findall('{http://www.mediawiki.org/xml/export-0.10/}page'):

        # Get the title and text elements for the page
        title = page.find('{http://www.mediawiki.org/xml/export-0.10/}title').text
        text = page.find('{http://www.mediawiki.org/xml/export-0.10/}revision/{http://www.mediawiki.org/xml/export-0.10/}text').text
        
        #Appending article to dict.
        allArticles[str(title)] = str(text)
    
    with open(cwd + '\\data\\allArticles.pickle', 'wb') as handle:
        pickle.dump(allArticles, handle, protocol=pickle.HIGHEST_PROTOCOL)



def processData():
    nlp = sp.load("en_core_web_lg")
    words = set(nltk.corpus.words.words())

    # Get path to current working directory
    cwd = os.getcwd()
    with open(cwd + '\\data\\allArticles.pickle', 'rb') as handle:
        allArticles = pickle.load(handle)
    i = 0
    print("Hmmmm", len(allArticles.keys()))
    for key in allArticles.keys():
        i += 1
        if i > 2:
            break
        
      #  print(f"key: {key}")
        processedText = allArticles[key]
       # processedText = nlp(allArticles[key])
      #  processedText = processedText.split("\n\n")
      #  processedText = processedText
        processedText = re.sub(r'[^A-Za-z0-9 ,.]+', '', processedText)
        processedText = st(processedText)
        tokenizedTexts = []
        tokenized_sents = [wt(i) for i in processedText]
       # tokenizedTexts.append(tokenized_sents)
        allArticles[key] = tokenized_sents
        for sent in processedText:
            tokenizedSent = str(wt(sent))
            processedText.remove(sent)
            processedText.append(tokenizedSent)
        

        allArticles[key] = processedText
          #  for w in wt(sent):
           #     if w not in words:
            #        if sent in processedText:
            #            processedText.remove(sent)
      #  print("LEN: ",len(tokenized_sents))  
      #  print("TEXT: ", tokenized_sents)
      ##  for word in processedText:
        #    word = re.sub(r'[^A-Za-z0-9 ,.]+', '', word)
         #   word = st(word)
          #  print("W: ", word)
       # print()
       # print(f"text: {processedText}")
    
    with open(cwd + '\\data\\processedArticles.pickle', 'wb') as handle:
        pickle.dump(allArticles, handle, protocol=pickle.HIGHEST_PROTOCOL)

#readData()
#processData()









#UTDATERT
    #xml_data = open(, encoding="utf8").read()  # Read data
    #xmlDict = xmltodict.parse(xml_data)  # Parse XML
    #xml_data = objectify.parse(cwd + '/data/simplewiki-articles.xml')  # Parse XML data
    #print(xml_data.)
    #i = 0
    #for key in xmlDict.keys():
    #   i += 1
    #  if i > 10:
    #     break
        #print(key)
    #root = xml_data.getroot()  # Root element

