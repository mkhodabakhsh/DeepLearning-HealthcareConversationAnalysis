def calc_probs_func(csv_path):
    import os
    import datetime
    import time
    from utilities import load_data, save_data, generate_probabilistic_embeddings
    from keras.models import load_model
    import spacy 
    from spacy.lang.en import English
    from swda import CorpusReader
    import pickle
    import math
    import pandas as pd 
    import unidecode
    import io


    resource_dir = 'data/'
    embeddings_dir = "embeddings/"
    model_dir = 'models/'
    model_name = 'Probabilistic Model'

    # Load metadata
    metadata = load_data(resource_dir + "metadata.pkl")
    word_frequency = 2
    frequency_data =load_data(embeddings_dir+'probabilistic_freq_'+str(word_frequency)+'.pkl')

    df = pd.read_csv(csv_path) 
    df.reset_index(drop=True, inplace=True)
    
    meta_labels = [i[0] for i  in metadata['labels']]
    test_comments = df['sentence']
    batch_name = 'all'
    a_dict = {}

    for t in range(len(test_comments)):  
        if "utterances" in a_dict:
            a_dict['utterances'].append(test_comments[t].split())
            a_dict['labels'].append('qw')
        else:
            a_dict["utterances"] = [test_comments[t].split()]
            a_dict['labels']=['qw']
    save_data(resource_dir + "sample_otter_data.pkl", a_dict)

    # Load Training and test sets
    test_data = load_data(resource_dir + 'sample_otter_data.pkl')
    utterances = test_data['utterances']
    labels = test_data['labels']

    test_x, test_y = generate_probabilistic_embeddings(test_data, frequency_data, metadata)
    val_data = load_data(resource_dir + 'val_data.pkl')

    # Parameters
    vocabulary_size = metadata['vocabulary_size']
    num_labels = metadata['num_labels']
    max_utterance_len = metadata['max_utterance_len']
    batch_size = 100
    hidden_layer = 128
    learning_rate = 0.001
    num_epoch = 10
    model_name = model_name + " -" + \
                " Epochs=" + str(num_epoch) + \
                " Hidden Layers=" + str(hidden_layer)

    model = load_model(model_dir + model_name + '.hdf5')

    test_scores = model.predict(test_x, batch_size=batch_size, verbose=0)
    
    for i , label in enumerate(meta_labels):
            df[label] = test_scores[...,i]

    cwd = os.getcwd()    
    os.chdir(cwd+'/IO_files')
    
    df.to_csv(cwd+'/IO_files' + '/sentence_list_plus_labels_plus_probs.csv')
    return 'Labels probability is calculated'