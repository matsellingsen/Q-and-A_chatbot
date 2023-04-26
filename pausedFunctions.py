def preprocessQuestionsAndLabels(filename):
   
    data = fetchData(filename)
    Q, L = data[0], data[2]

    print("Q: ", len(Q))
    print("L: ", len(L))

    #Remove stopwords
    i = 0
    processedQ = []
    compressedQ = []

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


   # encoder = KBinsDiscretizer(
   # n_bins=1000, encode="ordinal", strategy="uniform", random_state=0)
    decomposer = TruncatedSVD(n_components=1000, random_state=42)
    vocab = tokenizer.get_vocab()

    # Lowercase all the words in the vocabulary
    lowercase_vocabulary = {k.lower(): v for k, v in vocab.items()}
    #tokenizer.set_vocab(lowercase_vocabulary)

    spacyPipeline = sp.load("en_core_web_sm", disable=["tokenizer"])
   # spacyPipeline.add_pipe("negex", config={"ent_types":["PERSON","ORG"]})

    for question in Q:
        tokenized_q = tokenizer.tokenize(question)
        tokenized_q += extractFeatures(tokenized_q, spacyPipeline)
        processedQ.append(" ".join(tokenized_q))

    for Q in processedQ[:10]:
        print(Q) 

    matrix = CountVectorizer(vocabulary=lowercase_vocabulary)
    questionMatrices = matrix.fit_transform(processedQ).toarray()

    for q in questionMatrices[:5]:
        print("Q før kompresjon: ", q)

    compressedQ = decomposer.fit_transform(questionMatrices)
  
    # split train and test data
    X_train, X_test, y_train, y_test = train_test_split(compressedQ, L, test_size=0.2, random_state=42)

    # Store in pickle-file
    trainingAndTestingData = [X_train, X_test, y_train, y_test]
    storeData("words_stopwords_length_NER_POS_Data", trainingAndTestingData)
    
    return Q


def preprocessQuestions(filename):
    #Fetching questions and storing them in variable "questions"
    data = fetchData(filename)
    questions = data[0]

    processedQuestions = [wt(q) for q in questions]

    storeData("processedQuestions", processedQuestions)

def preprocessQuestionsAndLabels(filename):
   
    data = fetchData(filename)
    Q, L = data[0], data[2]

    print("Q: ", len(Q))
    print("L: ", len(L))
   # print("L: ", L)
    stemmer = PorterStemmer()
    #Tokenizing questions by words
    #Q = [wt(sent) for sent in Q]

    #Remove stopwords
    i = 0
    processedQ = []
    spacyPipeline = spacy.load("en_core_web_sm")

    tempQ = spacyPipeline(Q[0])
    test = []
    for token in tempQ:
        print("WORD: ", token)
        print("POS: ", token.tag_)
        print("NER: ", token.ent_type_)
        test.append(token.tag_)
        test.append(token.ent_type_)
    
    print("TEST1: ", test)

    for question in Q:
       # print(question)
     #   removelist = []
      #  for token in question:
       #      if token in sw.words("english"):
        #        removelist.append(token)
        #for e in removelist:
         #   question.remove(e)
    
        #Stemming
   #     for i in range(len(question)):
    #        question[i] = stemmer.stem(question[i])
        
        #Convert list of words to string of words to be used in Bag-Of-Words function. 

    #FINDING FEATURES
        #Doc-length
        docLength = len(question)

        #Spacy-pipeline
        tempQ = spacyPipeline(question)
        tempList = []
       # tempList.append(str(docLength))
        for token in tempQ:
            tempList.append(token.tag_)
            tempList.append(token.ent_type_)

       # processedQ.append(tempQ)

        processedQ.append(" ".join(tempList))
        
        i += 1
        print(i)
       # print(question)

    
    print("TEST: ", processedQ[:10])
  #  print("Q: ", Q[:10])
  #  print("ProcessedQ: ", processedQ[:10])
    matrix = CountVectorizer(max_features=10000)
    questionMatrices = matrix.fit_transform(processedQ).toarray()

    # split train and test data
    X_train, X_test, y_train, y_test = train_test_split(questionMatrices, L, test_size=0.2, random_state=42)

    # Store in pickle-file
    trainingAndTestingData = [X_train, X_test, y_train, y_test]
    storeData("trainingAndTesting_Data", trainingAndTestingData)
    
    return Q


def preprocessQuestionsAndLabels(filename):
   
    data = fetchData(filename)
    Q, L = data[0], data[2]

    print("Q: ", len(Q))
    print("L: ", len(L))

    #Remove stopwords
    i = 0
    processedQ = []
    compressedQ = []
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
   # encoder = KBinsDiscretizer(
   # n_bins=1000, encode="ordinal", strategy="uniform", random_state=0)
    decomposer = TruncatedSVD(n_components=1000, random_state=42)
    #vocab = tokenizer.get_vocab()


    for question in Q:
        tokenized_q =   tokenizer.tokenize(question)
        tokenized_q = extractFeatures(tokenized_q)
        processedQ.append(" ".join(tokenized_q))

    for Q in processedQ[:10]:
        print("HMMMMM: ", Q) 

    matrix = CountVectorizer()
    #questionMatrices = matrix.fit_transform(processedQ).toarray()

    for q in questionMatrices[:10]:
        print("Q før kompresjon: ", q)
        print("Lengde: ", len(q))

    #compressedQ = decomposer.fit_transform(questionMatrices)
  
    # split train and test data
    X_train, X_test, y_train, y_test = train_test_split(questionMatrices, L, test_size=0.2, random_state=42)

    # Store in pickle-file
    trainingAndTestingData = [X_train, X_test, y_train, y_test]
    storeData("lengthAndStopwordcount_Data", trainingAndTestingData)
    
    return Q

def trainModel(filename):

    data = fetchData(filename)

    X_train, X_test, Y_train, Y_test = data[0], data[1], data[2], data[3]

    model = MLPClassifier(activation="relu", solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(16, 3), random_state=1)
    trainedModel = model.fit(X_train, Y_train)
    predicted = trainedModel.predict(X_test)
    correct = 0
    for i in range(len(predicted)):
        if predicted[i] == Y_test[i]:
            correct += 1
       # print(f"result: {predicted[i]}:{test_Y[i]}")
    print(f"Accuracy: ", correct/len(predicted))
    print(f"f1-score: ", str(f1_score(Y_test, predicted, average=None)))


def findFeatures():
    questions = fetchData("processedQuestions")
    passages = fetchData("processedPassages")

    print(questions[0])
    print(passages[0])
    


def check(filename):
    data = fetchData(filename)
    labels = data[2]
    zeros = 0
    ones = 0
    for elem in labels:
        if elem == 0:
            zeros += 1
        else:
            ones += 1

    print("total: ", len(labels))
    print("zeros: ", zeros)
    print("ones: ", ones)


def check():
    data = fetchData("trainingAndTesting_Data")
    Y_train = data[2]
    Y_test = data[3]
    zeros_train = 0
    zeros_test = 0
    ones_train = 0
    ones_test = 0
    for elem in Y_train:
        if elem == 0:
            zeros_train += 1
        else:
            ones_train += 1
    for elem in Y_test:
        if elem == 0:
            zeros_test += 1
        else:
            ones_test += 1
    
    print(f"Zeros_train: {zeros_train/len(Y_train)}, ones_train: {ones_train/len(Y_train)}")
    print(f"zeros_test: {zeros_test/len(Y_test)}, onser_test: {ones_test/len(Y_test)}")
    #print("UNIKE TALL: ", list(unique))
    #print("MAX: ", max(list(unique)))
    #print("MIN: ", min(list(unique)))


def preprocessQuestions(filename):
    #Fetching questions and storing them in variable "questions"
    data = fetchData(filename)
    questions = data[0]

    processedQuestions = [wt(q) for q in questions]

    storeData("processedQuestions", processedQuestions)


def preprocessPassages(filename):

    #Fetching passages and storing them in variable "P"
    data = fetchData(filename)
    P = data[0][1]

   # Split into Header and Info, and transform all text to lowercase.
    splittedPassages = [elem.lower().split("--") for elem in P]
    headerP = [elem[0] for elem in splittedPassages]
    infoP = [elem[1] for elem in splittedPassages]

    #Sentence-tokenizing
    stPassages = [st(elem) for elem in infoP]

    processedPassages = []
    for doc in stPassages:
    # Remove punctuation
        processedPassage = [re.sub(r'[^\w\s]+', '', elem) for elem in doc]

    # tokenize
        processedPassage = [wt(elem) for elem in processedPassage]

    #Removing words containing non-english letters     
        for elem in processedPassage:
            removelist = [] #List for temporary storing words to be removed.
            for token in elem:
               # if token in sw.words("english"):
               #     removelist.append(token)
                if not re.match("^[a-zA-Z0-9]+$", token):
                    removelist.append(token)
                
            for e in removelist:
                elem.remove(e)
            processedPassages.append(processedPassage)

    #Storing processed passages in pickle-file
    storeData("processedPassages", processedPassages)
