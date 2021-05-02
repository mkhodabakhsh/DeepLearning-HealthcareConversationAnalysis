from __future__ import print_function
import os
import sys
import numpy as np
import pandas as pd
import keras
from sentence_types import load_encoded_data
from sentence_types import encode_data, import_embedding
from sentence_types import get_custom_test_comments
from keras.preprocessing import sequence
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer

from affirmation_function import affirmation_train_and_test
from reflection_function  import reflection_train_and_test
from calculate_probs_function import calc_probs_func
from generate_html_ import generate_html_function


# User can load a different model if desired
model_name      = "C:\\Users\Mohammad\Desktop\open_close_questions/models/2cnn"
embedding_name  = "C:\\Users\Mohammad\Desktop\open_close_questions/data/default"
load_model_flag = False
arguments       = sys.argv[1:len(sys.argv)]
if len(arguments) == 1:
    model_name = arguments[0]
    load_model_flag = os.path.isfile(model_name+".json")
print(model_name)
print("Load Model?", (load_model_flag))

# Model configuration
maxlen = 300
batch_size = 64
embedding_dims = 75
pool_size = 3
stride = 1
filters = 75
kernel_size = 5
epochs = 2
# Add parts-of-speech to data
pos_tags_flag = True


#Export & load embeddings
x_train, x_test, y_train, y_test = load_encoded_data(data_split=0.8, embedding_name=embedding_name,
                                                     pos_tags=pos_tags_flag)

word_encoding, category_encoding = import_embedding(embedding_name)
max_words   = len(word_encoding) + 1
num_classes = np.max(y_train) + 1
print(max_words, 'words')
print(num_classes, 'classes')
print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('Convert class vector to binary class matrix (for use with categorical_crossentropy)')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
load_model_flag = True
if not load_model_flag:
    print('Constructing model!')
    model = Sequential()
    model.add(Embedding(max_words, embedding_dims,
                        input_length=maxlen))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=stride))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Dropout(0.1))
    model.add(Conv1D(filters//2,
                     kernel_size//2 + 1,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
else:
    print('Loading model!')
    # load json and create model
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(model_name + ".h5")
    print("Loaded model from disk")
    # evaluate loaded model on test data
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)  ####>???????#####
test_comments, test_comments_category = get_custom_test_comments()
# 2: Statement (Declarative Sentence)    1: Question (Interrogative Sentence)
# 0: Exclamation (Exclamatory Sentence)  3: Command (Imperative Sentence)


current_dir = os.getcwd()
data = pd.read_csv(current_dir + '/sentence_list.csv') 
trans_otter = data["sentence"].tolist()
grnd_truth_oq,grnd_truth_cq = data["open_q"],data["close_q"]
grnd_truth_aff,grnd_truth_ref = data["affirm"],data["reflect"]

Qlist_n = []
Qstr_ = ""
for t in range(len(trans_otter)): 
    Qlist_n.append(trans_otter[t].rstrip("\n"))
    Qstr_ = Qstr_ + str(trans_otter[t])

num_ = len(Qlist_n)
test_comments , test_comments_category=[],[]
test_comments = Qlist_n
test_comments_category =  ['statement']*num_

#from termcolor import colored
#def write_red(f, str_):    f.write('<p style="color:#ff0000">%s</p>' % str_)
#def write_black(f, str_):  f.write('<p style="color:#000000">%s</p>' % str_)

real , test =[],[]
x_test, _, y_test, _ = encode_data(test_comments, test_comments_category, data_split=1.0,
                                   embedding_name=embedding_name, add_pos_tags_flag=pos_tags_flag)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
y_test = keras.utils.to_categorical(y_test, num_classes)
#score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)  ####  ?????? ######
#print('Manual test')
#print('Test accuracy:', score[1])

predictions = model.predict(x_test, batch_size=batch_size, verbose=0)
for i in range(0, len(predictions)):
    real.append(y_test[i].argmax(axis=0))
    test.append(predictions[i].argmax(axis=0))



############## GET INPUT PATH ##########
current_dir = os.getcwd()
user_input = input("Enter the name of your CSV file: ")
path_input = current_dir + '/'+ user_input
assert os.path.exists(path_input), "I did not find the file at, "+str(path_input)

calc_probs_func(path_input)#'C:\\Users\Mohammad/Desktop/affirmation_reflection/sentence_list_plus_labels.csv')

path_csv_test = current_dir + '/sentence_list_plus_labels_plus_probs.csv'


out_aff = affirmation_train_and_test(path_csv_test)
out_ref = reflection_train_and_test(path_csv_test)
index_aff,index_ref = out_aff[1],out_ref[1]

current_dir = os.getcwd()
data = pd.read_csv(current_dir + '/sentence_list_plus_labels_plus_probs.csv') 
trans_otter = data["sentence"].tolist()
grnd_truth_oq,grnd_truth_cq = data["open_q"],data["close_q"]
grnd_truth_aff,grnd_truth_ref = data["affirm"],data["reflect"]


question_flag = test
generate_html_function(test_comments,question_flag,grnd_truth_oq,grnd_truth_cq,grnd_truth_aff,grnd_truth_ref,index_aff,index_ref)