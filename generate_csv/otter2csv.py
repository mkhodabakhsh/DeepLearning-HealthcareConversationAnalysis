import numpy as np
import pandas as pd
import spacy 
from spacy.lang.en import English
from swda import CorpusReader
from utilities import *
from otter2csv import *
import sys
import os

def generate_otter_csv():
    
    user_input = input("Enter the path of your file: ")

    assert os.path.exists(user_input), "I did not find the file at, "+str(user_input)
    
    prob_True_reflect, prob_True_affirm, prob_total = [0]*41, [0]*41, [0]*41
    counter_reflect, counter_affirm = 0 , 0


    resource_dir = 'data/'
    embeddings_dir = "embeddings/"
    model_dir = 'models/'
    model_name = 'Probabilistic Model'


    df = pd.DataFrame([], index = []) 
    data = { "sentence": [], "open_q":   [], "close_q":  [], "affirm":   [], "reflect":  []}
    df = pd.DataFrame(data)


    resource_dir = 'data/'
    embeddings_dir = "embeddings/"
    model_dir = 'models/'
    model_name = 'Probabilistic Model'

    content_Otter = open(user_input, "r")
    time_Otter , trans_otter= [], []
    for index, str_ in enumerate(content_Otter):
        if index%3 == 1: trans_otter.append(str_)   

    Qlist_n = []
    Qstr_ = ""
    for t in range(len(trans_otter)): 
        Qlist_n.append(trans_otter[t].rstrip("\n"))
        Qstr_ = Qstr_ + str(trans_otter[t])   

    
    nlp = spacy.load("en_core_web_lg", has_vector=True)
    nlp.add_pipe(nlp.create_pipe('sentencizer'), first=True)
    doc = nlp(Qstr_)

    sw_remove, Qstr_n = 0 , []

    for sent in doc.sents:
        if sw_remove == 0:   Qstr_n.append(str(sent))
        elif sw_remove == 1:
            text_tokens = word_tokenize(str(sent))
            tokens_without_sw= [word for word in text_tokens if not word in all_stopwords]
            if tokens_without_sw:
                filtered_sentence = (" ").join(tokens_without_sw)
                Qstr_n.append(str(filtered_sentence))          
    num_ = len(Qstr_n)
    test_comments , test_comments_category=[],[]
    test_comments = Qstr_n
    test_comments_category =  ['statement']*num_


    # fill the datafrmae row by row
    for index_,sent in enumerate(test_comments):
        list_ = [test_comments[index_], 0, 0, 0, 0]
        df.loc[len(df.index)] = list_

    path, file = os.path.split(user_input)

    df.to_csv(path + '/sentence_list.csv')

    return "CSV file is generated."