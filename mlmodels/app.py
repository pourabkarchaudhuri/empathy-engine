# things we need for NLP
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

import pickle
import pandas as pd
import numpy as np
import json

import random
import entity_extraction
import sentiment_analysis
import stress_analysis


import os
THRESHOLD = 0.99
DEFAULT_STRESS = 78

stress = DEFAULT_STRESS
# print("INITIALIZING STRESS LEVEL : {}".format(stress))
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# Toggle comment above to run on CPU
import tensorflow as tf

# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.50
# set_session(tf.Session(config=config))
#----------------------------------------
# Toggle comments above to run on GPU Fraction

data = pickle.load( open( "keras-assistant-data.pkl", "rb" ) )
words = data['words']
classes = data['classes']

with open('intents.json') as json_data:
    intents = json.load(json_data)

extraction_dict = ["Okay. Fine. I'll come with you. Just get me to safety. Let's go!", "Then what are we waiting for. Let's go!"]
fallback_dict = ["Please ask me something else!", "I dont think you should be asking me that", "I am not going to respond to that", "I dont want to talk about that. If you have any other questions then ok."]
repeat_dict = ['stop repeating yourself!', "you are saying the same thing over and over again", "stop saying the same thing", "i am in so much pain, and here you are asking me the same thing over and over again!"]
def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    # print ("found in bag: %s" % w)
                    continue

    return(np.array(bag))


# p = bow("Load bood pessure for patient", words)
# print (p)
# print (classes)



# Use pickle to load in the pre-trained model
global graph
graph = tf.get_default_graph()

with open(f'keras-assistant-model.pkl', 'rb') as f:
    model = pickle.load(f)

def classify_local(sentence):
    ERROR_THRESHOLD = THRESHOLD
    
    # generate probabilities from the model
    input_data = pd.DataFrame([bow(sentence, words)], dtype=float, index=['input'])
    results = model.predict([input_data])[0]
    # filter out predictions below a threshold, and provide intent index
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], str(r[1])))
    # return tuple of intent and probability
    
    return return_list

def conversation_logger(input_sentence, intent_name, output_sentence):
    with open('conversations.txt', 'a') as file:
        file.write("INPUT : " + input_sentence + "\n" + "INTENT : " + intent_name + "\n" + "OUTPUT : " + output_sentence + "\n\n")

def fallback_logger(input_sentence):
    with open('fallback_sentences.txt', 'a') as file:
        file.write(input_sentence + "\n")

classify_local('Hello World!')

app = Flask(__name__)
CORS(app)

@app.route("/ml/api/v1.0/assistant", methods=['POST'])
def classify():
    global stress
    ERROR_THRESHOLD = THRESHOLD
    context = None
    sentence = request.json['sentence']
    # print(sentence)
    sentiment = sentiment_analysis.sentiment_analyzer(sentence)
    
    if "context" in request.json:
        # print("Context present : ", request.json['context'])
        context = request.json['context']

    else:
        # print("Context not present!")
        context = None

    entities = entity_extraction.named_entity_extraction(sentence)
    # generate probabilities from the model
    input_data = pd.DataFrame([bow(sentence, words)], dtype=float, index=['input'])
    results = model.predict([input_data])[0]
    # filter out predictions below a threshold

    # print("Before Filter : ", results)
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]

    # print("After Filter : ", results)
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    output_context = None
    # print("Results Length : ", len(results))
    if len(results) == 0:
        # print("Fallback detected")
        stress_payload = stress_analysis.stress_analyzer(sentiment['polarity'], 'fallback', stress)

        stress = stress_payload['stress']
        trigger = stress_payload['trigger']
        responsive = stress_payload['responsive']
        reaction = stress_payload['reaction']
        completion = stress_payload['completion']

        fallback_logger(sentence)
        return_list.append({"query": sentence, "intent": "fallback", "response": random.choice(fallback_dict), "context": None, "probability": "0.00", "entities": None, "sentiment":sentiment, "stress":stress, "trigger": trigger, "responsive":responsive, "reaction":reaction, 'completion':False})
       
        
    else:
        # print("Inference Exists")
        for r in results:
            if context != None:
                    classes[r[0]] = context
                    # print("Class Value for context: ", classes[r[0]])

            for x_tend in intents['intents']:
                
                if classes[r[0]] == x_tend['tag']:
                    # print("Entities Length : ", len(entities))
                    if x_tend['context'] == "":
                        output_context = None
                    if entities is None:
                        entities = None
                    elif len(entities) == 0:
                        entities = None
                    
                    stress_payload = stress_analysis.stress_analyzer(sentiment['polarity'], classes[r[0]], stress)

                    stress = stress_payload['stress']
                    trigger = stress_payload['trigger']
                    responsive = stress_payload['responsive']
                    reaction = stress_payload['reaction']
                    completion = stress_payload['completion']

                    if completion is True:
                        print("Extraction completion event triggered!")
                        return_list.append({"query": sentence, "intent": classes[r[0]], "response": random.choice(extraction_dict), "context": output_context, "probability": str(round(r[1],2)), "entities": entities, "sentiment":sentiment, "stress":stress, "trigger":trigger, "responsive":responsive, "reaction":reaction, 'completion':completion})

                    if reaction == 'extreme':
                        if stress_payload['repeat'] is not None:
                            return_list.append({"query": sentence, "intent": classes[r[0]], "response": random.choice(repeat_dict), "context": output_context, "probability": str(round(r[1],2)), "entities": entities, "sentiment":sentiment, "stress":stress, "trigger":trigger, "responsive":responsive, "reaction":reaction, 'completion':completion})
                          
                    elif reaction == 'shock':
                            return_list.append({"query": sentence, "intent": classes[r[0]], "response": "", "context": output_context, "probability": str(round(r[1],2)), "entities": entities, "sentiment":sentiment, "stress":stress, "trigger":trigger, "responsive":responsive, "reaction":reaction, 'completion':completion})
                          
                    normal_response = random.choice(x_tend['responses'])
                    normal_intent = classes[r[0]]

                    conversation_logger(sentence, normal_intent, normal_response)
                    return_list.append({"query": sentence, "intent": normal_intent, "response": normal_response, "context": output_context, "probability": str(round(r[1],2)), "entities": entities, "sentiment":sentiment, "stress":stress, "trigger":trigger, "responsive":responsive, "reaction":reaction, 'completion':completion})
        # return tuple of intent and probability

    response = jsonify({"result":return_list, "error":None})
    # print("Completion Status : {}".format(completion))
    if completion:
        stress = DEFAULT_STRESS
        # print("Level complete. Resetting Stress to default {}".format(DEFAULT_STRESS))
        
    # print("GLOBAL STRESS LEVEL : {}".format(stress))
    return response

# running REST interface, port=5000 for direct test, port=5001 for deployment from PM2
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000)

