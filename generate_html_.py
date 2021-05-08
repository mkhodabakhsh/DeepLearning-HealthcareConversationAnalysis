#  Generate HTML output 

#initialize the Matcher with a vocab. The matcher must always share the same vocab with the documents it will operate on.

def generate_html_function(test_comments,question_flag,grnd_truth_oeq,grnd_truth_ceq,grnd_truth_aff,grnd_truth_ref,index_aff,index_ref):
    import spacy 
    from spacy.lang.en import English
    from spacy.matcher import Matcher
    import random
    from tabulate import tabulate
    from termcolor import colored
    from sklearn.metrics import confusion_matrix
    ###############################################
    from spacy.matcher import Matcher

    #initialize the Matcher with a vocab. The matcher must always share the same vocab with the documents it will operate on.
    nlp = spacy.load("en_core_web_sm")
    matcher = Matcher(nlp.vocab)

    #Sentence patterns using question word clauses (Subject + verb + interrogative adverb / pronoun + clause)
    #I know where he lives.
    #He asked when he should come.
    #I wonder why he is late today.
    #She explained how it could be done.

    pattern1 = [{"LOWER": "what"}, {"POS": "AUX"}]
    pattern2 = [{"LOWER": "how"}, {"POS": "AUX"}]
    pattern3 = [{"LOWER": "who"}, {"POS": "AUX"}]
    pattern4 = [{"LOWER": "where"}, {"POS": "AUX"}]
    pattern5 = [{"LOWER": "when"}, {"POS": "AUX"}]

    pattern6 = [{"LOWER": "what"}, {"POS": "VERB"}]
    pattern7 = [{"LOWER": "how"}, {"POS": "VERB"}]
    pattern8 = [{"LOWER": "who"}, {"POS": "VERB"}]
    pattern9 = [{"LOWER": "where"}, {"POS": "VERB"}]
    pattern10 = [{"LOWER": "when"}, {"POS": "VERB"}]

    pattern11 = [{"LOWER": "what"},  {"POS": "ADP"}]
    pattern12 = [{"LOWER": "how"},  {"POS": "ADP"}]
    pattern13 = [{"LOWER": "who"},  {"POS": "ADP"}]
    pattern14 = [{"LOWER": "where"},  {"POS": "ADP"}]
    pattern15 = [{"LOWER": "when"},  {"POS": "ADP"}]

    pattern16 = [{"LOWER": "what"}, {}, {"POS": "AUX"}, {"POS": "PRON"}]
    pattern17 = [{"LOWER": "how"}, {}, {"POS": "AUX"}, {"POS": "PRON"}]
    pattern18 = [{"LOWER": "who"}, {}, {"POS": "AUX"}, {"POS": "PRON"}]
    pattern19 = [{"LOWER": "where"}, {}, {"POS": "AUX"}, {"POS": "PRON"}]
    pattern20 = [{"LOWER": "when"}, {}, {"POS": "AUX"}, {"POS": "PRON"}]
    matcher.add("WH_Q", None, pattern1,  pattern2,  pattern3, pattern4,  pattern5,  pattern6, pattern7,  pattern8,  pattern9, pattern10,  pattern11,  pattern12,  pattern13, pattern14,  pattern15,  pattern16, pattern17,  pattern18,  pattern19, pattern20)
    #####################################################################################
    
    def write_red(f, str_):    f.write('<p style="color:#FF3333">%s</p>' % str_)
    def write_black(f, str_):  f.write('<p style="color:#000000">%s</p>' % str_)
    def write_blue(f, str_):  f.write('<p style="color:#0080ff">%s</p>' % str_)
    def write_blue_bold(f, str_):  f.write('<b style="color:#0000ff">%s</b>' % str_)
    def write_black_bold(f, str_):  f.write('<b style="color:#000000">%s</b>' % str_)
    def write_orange(f, str_):  f.write('<p style="color:#ff9933">%s</p>' % str_)
    def write_green(f, str_):  f.write('<p style="color:#00ff80">%s</p>' % str_)
    def write_header3(f, str_):  f.write('<h3 style="text-align:center">%s</h3>' % str_)
    def write_table_text(f, str_):  f.write('<h4 style="text-align:center">%s</h4>' % str_)
    def write_header5(f, str_):  f.write('<h5 style="text-align:left;color:#0000ff">%s</h5>' % str_)

    Count_=0
    index_oeq, index_ceq=[0]*len(question_flag),[0]*len(question_flag)

    count_ceq , count_oeq = 0 ,0
    count_aff , count_ref = sum(index_aff) , sum(index_ref)
    example_oeq, example_ceq, example_aff, example_ref = [],[],[],[]
    
    for ind_ , value_ in enumerate(index_aff):
        if value_==1: example_aff.append(test_comments[ind_]) 
    for ind_ , value_ in enumerate(index_ref):
        if value_==1: example_ref.append(test_comments[ind_]) 
    
    for t , i in enumerate(test_comments):  
        if question_flag[t]==1:
            doc = nlp(i)
            matches = matcher(doc)
            if matches == []:
                index_ceq[t]=1
                count_ceq+=1
                example_ceq.append(test_comments[t])
            else:
                index_oeq[t]=1
                for match_id, start, end in matches:                
                    count_oeq+=1
                    example_oeq.append(test_comments[t])
                    Count_+=1
        else:
            pass
    


    f = open('results/test_platform.html', 'w')
    f.write('<html>')
    f.write('<h1 style="text-align:center">Color Labeled Transcript</h1>')
    f.write('<hr style="height:30px;border-width:0;color:white;"background-color:white></h1>')


    ######   TABLE#1 #######
    write_header3(f, '           Total number of sentences:    '+ '<b style="color:#0000ff">%s</b>' % str(len(test_comments)))

    count_open, count_close, count_affirm, count_reflect  = count_oeq, count_ceq, 0, 0
    percent_oeq, percent_ceq, percent_aff, percent_ref = count_oeq/len(test_comments)*100, count_ceq/len(test_comments)*100, count_aff/len(test_comments)*100, count_ref/len(test_comments)*100
    class_list = ['Open-Ended question','Closed-Ended question','Affirmation statement','Reflection statement']
    index_list_oeq, index_list_ceq = range(len(example_oeq)), range(len(example_ceq))
    index_list_aff, index_list_ref = range(len(example_aff)), range(len(example_ref))

    rndsmpl_oeq, rndsmpl_ceq = random.sample(index_list_oeq, min(3,len(index_list_oeq))), random.sample(index_list_ceq,min(3,len(index_list_ceq)))
    rndsmpl_aff, rndsmpl_ref = random.sample(index_list_aff, min(3,len(index_list_aff))), random.sample(index_list_ref, min(3,len(index_list_ref)))

    table_ = [["CLASS",'OCCURRENCE NUMBER','EXAMPLES'],
    ["    ","     ","   "],["    ","     ","   "],["    ","     ","   "],["","",example_oeq[rndsmpl_oeq[0]]],
    [class_list[0],str(count_open)+'('+str("{:.1f}".format(percent_oeq))+'%)',example_oeq[rndsmpl_oeq[1]]],
    ["","",example_oeq[rndsmpl_oeq[-1]]],
    ["    ","     ","____________________________________________________________________________________________________"],["    ","     ","   "],["    ","     ","   "],

    ["","",example_ceq[rndsmpl_ceq[0]]],
    [class_list[1],str(count_close)+'('+str("{:.1f}".format(percent_ceq))+'%)',example_ceq[rndsmpl_ceq[1]]],
    ["","",example_ceq[rndsmpl_ceq[-1]]],
    ["    ","     ","____________________________________________________________________________________________________"],["    ","     ","   "],["    ","     ","   "],

    ["","",example_aff[rndsmpl_aff[0]]],
    [class_list[2],str(count_aff)+'('+str("{:.1f}".format(percent_aff))+'%)',example_aff[rndsmpl_aff[1]]],
    ["","",example_aff[rndsmpl_aff[-1]]],
    ["    ","     ","____________________________________________________________________________________________________"],["    ","     ","   "],["    ","     ","   "],

    ["","",example_ref[rndsmpl_ref[0]]],
    [class_list[3],str(count_ref)+'('+str("{:.1f}".format(percent_ref))+'%)',example_ref[rndsmpl_ref[1]]],
    ["","",example_ref[rndsmpl_ref[-1]]],["    ","     ","   "],["    ","     ","   "]]

    table_cnotent_ = '<style>table{border-top: 1px solid gray;border-bottom: 1px solid gray;margin-left:auto;argin-top:12px;margin-right:auto;display: inline-block;border=1 frame=hsides rules=rows>}</style>'+tabulate(table_ , headers="firstrow", tablefmt="html" , numalign=("center"), stralign=("center"))

    write_table_text(f,table_cnotent_)
    f.write('<hr style="height:10px;border-width:0;color:white;background-color:white">')

    #### END OF TABLE #1  ######
    f.write('<hr style="height:15px;border-width:0;color:white;"background-color:white></h1>')
    f.write('<hr style="height:15px;border-width:0;color:rgba(192, 192, 191, 0.3);background-color:rgba(192, 192, 191, 0.15)">')
    f.write('<hr style="height:0px;border-width:0;color:white;"background-color:white></h1>')
    f.write('<hr style="height:10px;border-width:0;color:white;background-color:white">')

    f.write('<h5 style="height:8px;border-width:0;color:rgba(0, 128, 255, 0.5);"background-color:rgba(255, 99, 71, 0.2);">OPEN ENDED QUESTION     </h5>')
    f.write('<h5 style="height:8px;border-width:0;color:rgba(255, 51, 51, 0.5);"background-color:rgba(255, 99, 71, 0.2);">CLOSED ENDED QUESTION</h5>')
    f.write('<h5 style="height:8px;border-width:0;color:rgba(0, 255, 128, 0.5);"background-color:rgba(255, 99, 71, 0.2);">AFFIRMATION</h5>')
    f.write('<h5 style="height:8px;border-width:0;color:rgba(255, 180, 41, 0.6);"background-color:rgba(255, 99, 71, 0.2);">REFLECTION</h5>')
    f.write('<hr style="height:15px;border-width:0;color:white;"background-color:white></h1>')
    f.write('<hr style="height:5px;border-width:0;color:white;"background-color:white></h1>')

    for t , i in enumerate(test_comments):  
        if question_flag[t]==1:
            doc = nlp(i)
            matches = matcher(doc)
            if matches == []:
                write_red(f, test_comments[t])
            else:
                for match_id, start, end in matches:
                    string_id = nlp.vocab.strings[match_id]  
                    span = doc[start:end]                    
                    write_blue(f, test_comments[t])
                    #print(string_id)
        elif index_aff[t]==1:
            write_green(f, test_comments[t])
        elif index_ref[t]==1:
            write_orange(f, test_comments[t])
        else:
            write_black(f, test_comments[t])
    
    f.write('<hr style="height:5px;border-width:0;color:white;"background-color:white></h1>')

    def calc_perf(FP_,FN_,TP_):
        if TP_ + FP_:
            prec_      = TP_ / (TP_ + FP_)
        else: prec_=0
        if TP_ + FN_:
            recall_    = TP_ / (TP_ + FN_)
        else: recall_=0
        try:
            if prec_+recall_:
                F1score_  = 2*prec_*recall_/(prec_+recall_)
            else: F1score_=0
        except:F1score_=0
        
        return prec_,recall_,F1score_

    ######   TABLE #2 #######
    f.write('<hr style="height:15px;border-width:0;color:white;"background-color:white></h1>')
    f.write('<hr style="height:0px;border-width:0;color:white;"background-color:white></h1>')
    write_header3(f, 'Model Performance'   )

    count_open, count_close, count_affirm, count_reflect  = count_oeq, count_ceq, 0, 0
    percent_oeq, percent_ceq, percent_aff, percent_ref = count_oeq/len(test_comments)*100, count_ceq/len(test_comments)*100, 0, 0
    class_list = ['Open-Ended question','Closed-Ended question','Affirmation statement','Reflection statement']

    TN_oeq,FP_oeq,FN_oeq,TP_oeq = confusion_matrix(grnd_truth_oeq, index_oeq, labels=[0,1]).ravel()
    TN_ceq,FP_ceq,FN_ceq,TP_ceq = confusion_matrix(grnd_truth_ceq, index_ceq, labels=[0,1]).ravel()
    TN_aff,FP_aff,FN_aff,TP_aff = confusion_matrix(grnd_truth_aff, index_aff, labels=[0,1]).ravel()
    TN_ref,FP_ref,FN_ref,TP_ref = confusion_matrix(grnd_truth_ref, index_ref, labels=[0,1]).ravel()

    prec_oeq,racall_oeq,F1score_oeq = calc_perf(FP_oeq,FN_oeq,TP_oeq)
    prec_ceq,racall_ceq,F1score_ceq = calc_perf(FP_ceq,FN_ceq,TP_ceq)
    prec_aff,racall_aff,F1score_aff = calc_perf(FP_aff,FN_aff,TP_aff)
    prec_ref,racall_ref,F1score_ref = calc_perf(FP_ref,FN_ref,TP_ref)

    str("{:.1f}".format(percent_oeq))+'%'
    table_ = [[str("CLASS"),"","","","","","TN","","","","","","FP","","","","","",'FN',"","","","","",'TP',"","","","","",'Precesion',"","","","","",'Recall',"","","","","",'F1-score'],
    ["","","","","","","",""],["","","","","","","",""],["","","","","","","",""],
    [class_list[0],"","","","","",TN_oeq,"","","","","",FP_oeq,"","","","","",FN_oeq,"","","","","",TP_oeq,"","","","","",str("{:.1f}".format(prec_oeq*100))+'%',"","","","","",str("{:.1f}".format(racall_oeq*100))+'%',"","","","","",str("{:.1f}".format(F1score_oeq*100))+'%'],
    ["","","","","","","",""],["","","","","","","",""],["","","","","","","",""],
    [class_list[1],"","","","","",TN_ceq,"","","","","",FP_ceq,"","","","","",FN_ceq,"","","","","",TP_ceq,"","","","","",str("{:.1f}".format(prec_ceq*100))+'%',"","","","","",str("{:.1f}".format(racall_ceq*100))+'%',"","","","","",str("{:.1f}".format(F1score_ceq*100))+'%'],
    ["","","","","","","",""],["","","","","","","",""],["","","","","","","",""],
    [class_list[2],"","","","","",TN_aff,"","","","","",FP_aff,"","","","","",FN_aff,"","","","","",TP_aff,"","","","","",str("{:.1f}".format(prec_aff*100))+'%',"","","","","",str("{:.1f}".format(racall_aff*100))+'%',"","","","","",str("{:.1f}".format(F1score_aff*100))+'%'],
    ["","","","","","","",""],["","","","","","","",""],["","","","","","","",""],
    [class_list[3],"","","","","",TN_ref,"","","","","",FP_ref,"","","","","",FN_ref,"","","","","",TP_ref,"","","","","",str("{:.1f}".format(prec_ref*100))+'%',"","","","","",str("{:.1f}".format(racall_ref*100))+'%',"","","","","",str("{:.1f}".format(F1score_ref*100))+'%']]

    table_cnotent_ = '<style>table{border-top: 1px solid gray;border-bottom: 1px solid gray;margin-left:auto;argin-top:12px;margin-right:auto;display: inline-block;border=1 frame=hsides rules=rows>}</style>'+tabulate(table_ , headers="firstrow", tablefmt="html" , numalign=("center"), stralign=("center"))
    write_table_text(f,table_cnotent_)
    f.write('<hr style="height:10px;border-width:0;color:white;background-color:white">')

    #### END OF TABLE #2  ######

    f.write('<hr style="height:15px;border-width:0;color:white;"background-color:white></h1>')
    f.write('<hr style="height:15px;border-width:0;color:rgba(192, 192, 191, 0.3);background-color:rgba(192, 192, 191, 0.15)">')
    f.write('<h5 style="height:5px;border-width:0;color:rgba(0, 0, 0, 0.8);"background-color:rgba(192, 192, 191, 0.15);">Created by NEMI.ai</h5>')
    f.write('<hr style="height:0px;border-width:0;color:white;"background-color:white></h1>')


    f.write('</html>')
    f.close()