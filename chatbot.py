import json
import random
import numpy as np
import os
import pickle
import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,Activation,Dropout
from tensorflow.keras.optimizers import SGD

IGNORE_WORDS = ["!","?",",","."]
json_file = "intents.json"

lemmatizer = WordNetLemmatizer()

# main loop
def main():
    def train_mode():
        new_tag = input("Please enter a tag: ")
        target = input("Please enter keywords or sentence: ")
        response = input("Please enter response: ")
        confirm(new_tag,target,response)
    
    def confirm(new_tag,target,response):
        print(f"""
    tag : {new_tag}
    keywords: {target}
    response: {response}
                """)
        final = input("Are you sure you want to train with this data? (y/n): ")
        if final.lower() == "y":
            print("data is trained")
            get_json(new_tag,target,response)
            training = update_pickle(json_file)
            train_model(training)
            train_mode()
        elif final.lower() == "n":
                print("Have another idea?")
                train_mode()
        else:
            confirm()        
            
    print("1. Train mode")
    print("2. Chat mode")
    mode = input("Please enter the number to choose mode: ")
    if mode == "1":
        train_mode()  
        
    elif mode == "2":
        chatbot()
    else:
        main()

#turn input into json
def get_json(new_tag,target,response):
    words = []
    classes = []
    word_list = nltk.word_tokenize(target)
    word_list = [lemmatizer.lemmatize(word.lower()) for word in word_list if word not in IGNORE_WORDS]
    
    if new_tag in classes:
        words[classes.index(new_tag)].extend(word_list)
    else:
        classes.append(new_tag)
        words.append(word_list)   
        
    if not os.path.exists(json_file):
        with open(json_file,"w") as f:
            json.dump({"intents":[]},f)
    
    data = json.loads(open(json_file).read())
    tags = [x["tag"] for x in data["intents"]]
    
    #backup the old data
    with open ("backup.json","w") as backup:
        json.dump(data,backup,indent=2)
        
    #update json    
    with open(json_file,"w") as f:
    
        for i in range(len(classes)):
            tag = classes[i]
            pattern = words[i]
            
            if tag in tags:
                loc = tags.index(tag)
                old_pattern = data["intents"][loc]["patterns"]
                if pattern !="":
                    old_pattern.extend(pattern)
                    new = sorted(set(data["intents"][loc]["patterns"]))
                    data["intents"][loc]["patterns"] = new
                
                if response != "":
                    data["intents"][loc]["response"].append(response)
                    new_res = sorted(set(data["intents"][loc]["response"]))
                    if "No response now" in new_res:
                        new_res.remove("No response now")
                    data["intents"][loc]["response"] = new_res

                
            else:
                if response == "":
                    response = "No response now"               
                data["intents"].append({"tag":tag,"patterns":pattern,"response":[response]})

            f.seek(0)
            json.dump(data,f,indent=2)
        print("json file is updated")
            
def update_pickle(json_file=json_file):
    words =[]
    classes =[]
    documents =[]
    with open (json_file,"r") as f:
        data = json.load(f)
    
    for intent in data["intents"]:
        words.extend(intent["patterns"])
        classes.append(intent["tag"])
        documents.append((intent["tag"],intent["patterns"]))
    
    pickle.dump(words,open("words.pkl","wb"))
    pickle.dump(classes,open("classes.pkl","wb"))
          
#change the data to binary since neural network only understand 0 and 1
    training = []
    output_empty = [0] * len(classes)
    
    for document in documents:
        bag = []
        word_patterns = document[1]
        
        for word in words:
            bag.append(1) if word in word_patterns else bag.append(0)
        
        output_row = output_empty[:]
        output_row[classes.index(document[0])]=1

        training.append([bag,output_row])
    return training

#train model
def train_model(training):
    training = np.array(training,dtype=object)
 
    train_x = list(training[:,0])
    train_y = list(training[:,1])

    model = Sequential()
    model.add(Dense(128,input_shape=(len(train_x[0]),),activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64,activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]),activation="softmax"))

    sgd = SGD(learning_rate=0.01,decay=1e-6,momentum=0.9,nesterov=True)
    model.compile(loss="categorical_crossentropy",optimizer=sgd,metrics=["accuracy"])

    hist = model.fit(np.array(train_x),np.array(train_y),epochs=200,batch_size=5,verbose=1)
    model.save("chatbot_model.h5",hist)
        
    print("-----Training Done-----")
    
def chatbot():
    words = pickle.load(open("words.pkl","rb"))
    classes = pickle.load(open("classes.pkl","rb"))
    model = load_model("chatbot_model.h5")
    
    #tokenize the input into list
    def clean_up_sentence(sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
        return sentence_words

    #change the tokenized list into binary
    def bag_of_words(sentence):
        sentence_words = clean_up_sentence(sentence)
        bag = [0] * len(words)
        for w in sentence_words:
            for i,word in enumerate(words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)

    #predict 
    def predict_class(sentence):
        bow = bag_of_words(sentence)
        res = model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        
        results.sort(key=lambda x:x[1],reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent":classes[r[0]],"probability":str(r[1])})

        return return_list

    def get_response(intents_list,intents_json):
        data = json.loads(open(intents_json).read())
        tag = intents_list[0]["intent"]
        list_of_intents = data["intents"]
        for i in list_of_intents:
            if i["tag"] == tag:
                result = random.choice(i["response"])
                break
        return result

    print("Bot is ready")

    while True:
        message = input("You: ")
        ints = predict_class(message)
        res = get_response(ints, json_file)
        print("Bot: ",res)
    

if __name__ == "__main__":

    main()        
            