from __future__ import print_function
import os
import sys
import numpy as np
import pandas as pd
from affirmation_function import affirmation_train_and_test
from reflection_function  import reflection_train_and_test
from calculate_probs_function import calc_probs_func
from generate_html_ import generate_html_function
from question_classification_function import question_classification_func

############## GET INPUT PATH ##########
owd = os.getcwd()

input_type = input("What is the type of the input file? Enter 1 for text file, 2 for CSV file. ")
print('your input is',input_type)
if int(input_type)==1:
    from otter2csv import *
    test_comments = generate_otter_csv(owd)
    user_input = 'sentence_list.csv'
elif int(input_type)==2:
    user_input = input("Enter the name of the CSV file: ")
else:
    print('Your input is not acceptable.')

cwd = os.getcwd()
save_dir = cwd
assert os.path.exists(save_dir + '/' + user_input), f"The CSV file doesn't exist at {str(save_dir)}"

os.chdir(owd)
cwd = os.getcwd()

out_qcf = question_classification_func(cwd)

test_comments , test = out_qcf[0],out_qcf[1]

calc_probs_func(save_dir + '/' + user_input)
path_csv_test = save_dir + '/sentence_list_plus_labels_plus_probs.csv'
out_aff = affirmation_train_and_test(owd, path_csv_test,test_comments)
out_ref = reflection_train_and_test(owd, path_csv_test)
index_aff,token_existence,index_ref = out_aff[1],out_aff[2],out_ref[1]

data = pd.read_csv(save_dir + '/sentence_list_plus_labels_plus_probs.csv') 
trans_otter = data["sentence"].tolist()
grnd_truth_oq,grnd_truth_cq = data["open_q"],data["close_q"]
grnd_truth_aff,grnd_truth_ref = data["affirm"],data["reflect"]

question_flag = test
generate_html_function(test_comments,question_flag,grnd_truth_oq,grnd_truth_cq,
                       grnd_truth_aff,grnd_truth_ref,index_aff,index_ref,token_existence)
os.remove(save_dir + '/sentence_list_plus_labels_plus_probs.csv')
