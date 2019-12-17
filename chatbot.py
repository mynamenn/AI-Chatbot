import nltk
import tflearn
import pickle
import random
import json
import numpy as np
from nltk.stem.lancaster import LancasterStemmer

nltk.download('punkt')
stemmer= LancasterStemmer()

with open("messages.json") as file:
    data= json.load(file)

try:
    with open("data.pickle", "wb") as f:
        words, labels, training, output= pickle.load(f)
except:
    words= []    #contains all the sentences in patterns
    labels= []   #contains tags
    docs_x= []   #contains words of setence
    docs_y= []   #store the tag for each word in doc_x to access it later

    for intent in data["intents"]:
        for patterns in intent["patterns"]:
            wrds= nltk.word_tokenize(patterns)   #split string to substring.
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words= [stemmer.stem(w.lower()) for w in words if w!='?']  #stem each word to its root (removing uneccesary extensions of word)
    words= sorted(list(set(words)))

    training=[]  #2D array containing 1 and 0 for each pattern
    output=[]
    indexIndicator=[0 for _ in range(len(labels))]  #[0,0,0,0] if 1 means there are words in that label
    for x,doc in enumerate(docs_x):
        bag=[]
        wrds=[stemmer.stem(w) for w in doc]
        for w in words:
            if w in wrds:
                bag.append(1)    #Neural network can only accept integer
            else:
                bag.append(0)
        indicator= indexIndicator[:]

        indicator[labels.index(docs_y[x])]=1

        training.append(bag)
        output.append(indicator)

    training= np.array(training)
    output= np.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

net= tflearn.input_data(shape=[None, len(words)])
net= tflearn.fully_connected(net, 8, activation= "relu")
net= tflearn.fully_connected(net, 8, activation= "relu")
net= tflearn.fully_connected(net, len(output[0]), activation="softmax")
net= tflearn.regression(net)
model = tflearn.DNN(net)

try:
    model.load_model("model.tflearn")          #load pretrained model
except:
    model = tflearn.DNN(net)
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bagOfWords(sentence, words):            #function turns a sentence into an array of words
    bag= [0 for _ in range(len(words))]

    s_words= nltk.word_tokenize(sentence)
    s_words= [stemmer.stem(word.lower()) for word in s_words]

    for i in s_words:
        for index, j in enumerate(words):
            if j==i:
                bag[index]=1
    return np.array(bag)

def chat():
    print("Start talking to the bot. (Ask me anything about my restaurant)")
    while True:
        inp= input("You: ")
        if inp.lower()== "quit":
            break

        result= model.predict([bagOfWords(inp, words)])
        result_max= np.argmax(result) #returns index of the highest prob

        tag= labels[result_max]
        if result[0][result_max]>0.8:            #if probability> 0.8, then print message.
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses= tg['responses']
            print(random.choice(responses))

        else:
            print("Sorry, I don't understand what you're saying. ")

chat()
