from fastapi import FastAPI
import tensorflow as tf
import json
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from os import path
from keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]

@app.get("/isTokenizerVocabTrainingDataAvailable")
def isTokenizerVocabTrainingDataAvailable():
    tokenizer_vocab_path = path.relpath("tokenizer/tokenizer_vocab.txt")
    if(os.path.isfile(tokenizer_vocab_path)):
        return True
    else:
        return False
    
@app.get("/isTrainedModelAvailable")
def isTrainedModelAvailable():
    model_path = path.relpath("models/my_model.h5")
    if(os.path.isfile(model_path)):
        return True
    else:
        return False

@app.post("/CMSAI/predictNextFourWords")
def generatesqlfromtext(input:str):
# Load tokenizer vocabulary
    tokenizer = Tokenizer()
    tokenizer_vocab_path = path.relpath("tokenizer/tokenizer_vocab.txt")
    with open(tokenizer_vocab_path,"r", encoding="utf-8") as f:
     for line in f:
        word, index = line.strip().split("\t")
        tokenizer.word_index[word] = int(index)

    model_path = path.relpath("models/my_model.h5")
    myModel = load_model(model_path)

    for i in range(4):
        token_text = tokenizer.texts_to_sequences([input])[0]
        #TODO: 41 is the max length of input sequences, this needs to be handled better, currently for demo this is hardcoded 
        token_list = pad_sequences([token_text], maxlen = 41, padding = 'pre')
  
        pos = np.argmax(myModel.predict(token_list, verbose = 0))

        for word, index in tokenizer.word_index.items():
            if index == pos:
                input = input +" " + word
    return input
