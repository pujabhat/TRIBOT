from django.shortcuts import render, HttpResponse
from django.http import JsonResponse
import json,pickle
import numpy 
import nltk
import tflearn
from nltk.stem.lancaster import LancasterStemmer
import random
import tensorflow
from chatbot.models import Question
from tensorflow.python.framework import ops

def load_model(training,output):
    ops.reset_default_graph()
    input_layer = tflearn.input_data(shape=[None, len(training[0])]) #46
    dense1 = tflearn.fully_connected(input_layer, 8)
    softmax = tflearn.fully_connected(dense1, len(output[0]), activation="softmax")
    net = tflearn.regression(softmax)
    model = tflearn.DNN(net)
    model.load("C:\Users\ACER\Downloads\Code_Part\Code_Part\Tribot_Chatbot\Tribot\demomodel2.tflearn")
    print(len(training[0]),len(output[1]))
    return model


def home(request):
    return render(request,'chatbot/index.html')
    
def message(request):
    if request.method=='POST' and request.headers.get('x-requested-with') == 'XMLHttpRequest':
        msg=request.POST.get('msg')
        print(msg)
        rs=chat(msg)
        data = {
            'msg':rs
        }
        print(msg,rs)
        q=Question(question_text=msg,reply=data['msg'])
        q.save()

        return JsonResponse(data)
    return JsonResponse({'msg':'can not get'})

def bag_of_words(s, words):
    with open("C:\Users\ACER\Downloads\Code_Part\Code_Part\Tribot_Chatbot\Tribot\stopwords.pickle", "rb") as f:
        stopwords =pickle.load(f)
    
    stemmer = LancasterStemmer()
    bag = []
    print(s)
    wrds=nltk.word_tokenize(s)
    print(wrds)
    wrds=[word.lower() for word in wrds if word.isalpha()]
    print(wrds)
    wrds = [word for word in wrds if word not in stopwords ]
    print(wrds)
    wrds=[stemmer.stem(word)for word in wrds ]
    print(wrds)

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)            
            
    print(numpy.array(bag))  
    return numpy.array(bag)




def chat(msg):
    with open("C:\Users\ACER\Downloads\Code_Part\Code_Part\Tribot_Chatbot\Tribot\chatbot.json") as file:
        data = json.load(file)
   
    with open("C:\Users\ACER\Downloads\Code_Part\Code_Part\Tribot_Chatbot\Tribot\data2.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
    model=load_model(training,output)
    results = model.predict([bag_of_words(msg, words)])
    print(results)
    results_index = numpy.argmax(results)
    if results[0][results_index] < 0.5:
        return "I apologize, I can not understand your question. Can you please provide more details or rephrase it for me?"
    
    tag = labels[results_index]

    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']
            return random.choice(responses)

        


