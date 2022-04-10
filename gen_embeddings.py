import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from bertopic import BERTopic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
def generate_matrices(model,X,word_to_index, words_per_topic = 10,nr_topics_1 = None):
    ### get target topics and probabilities of each document topic pair

    topics,probs = model.transform(X)

    ### calclate remaining topic probability as 1 - sum of given topic probabilities.
    document_to_topic_bas = probs
    document_to_topic = []
    for i in range(len(document_to_topic_bas)):
        sum1 = 0.0
        for j in document_to_topic_bas[i]:
            sum1 = sum1 + j
        templ = [max(1-sum1,0.0)]
        for j in document_to_topic_bas[i]:
            templ.append(j)
        document_to_topic.append(templ)



    #print(topics[0] , document_to_topic[0])

    vocab_len = len(list(word_to_index.items()))

    topic_to_word_dict  = model.get_topics()
    topic_len = len(list(topic_to_word_dict.items()))

    ## initialize topic_to_word matrix
    topic_to_word = []
    for i in range(topic_len):
        temparr = []
        for j in range(vocab_len):
            temparr.append(0)
        topic_to_word.append(temparr)


    for i in topic_to_word_dict:
        temparr = topic_to_word_dict[i]
        size = len(temparr)
        for j in range(size):
            if(temparr[j][0] in word_to_index):
                 ## make sure that -1 corresponds to 0
                topic_to_word[i+1][word_to_index[temparr[j][0]]] = temparr[j][1]

    ## convert to pytorch tensors.
    document_to_topic_tens = torch.tensor(document_to_topic)
    topic_to_word_tens= torch.tensor(topic_to_word)

    return (document_to_topic_tens,topic_to_word_tens)





def get_word_pos():
    word_to_index = dict()
    embedding_dictionary = dict()
    count0 = 0
    with open('glove.6B.100d.txt' , 'r',encoding = 'utf-8') as f1:
        for line in f1:
            lis = line.split()
            word_to_index[lis[0]] = count0
            templist = []
            for index in range(1,len(lis)):
                templist.append(float(lis[index]))
            embedding_dictionary[lis[0]] = templist
            count0 = count0+1
    return (word_to_index,embedding_dictionary)




def parse_df_method_concat(inp,words_from_head,words_from_body,embedding_dict,topic2word,word2index):
    final_feature = []
    topic2wordnum = topic2word.numpy()
    el = len(embedding_dict[list(embedding_dict.items())[0][0]])
    tl  = topic2wordnum.shape[0]

    for headlines,body in zip(inp['Headline'].values,inp['articleBody'].values):
        record = []
        c0 = 0
        for word in headlines:
            temp = []
            if(word in embedding_dict):

                for j in embedding_dict[word]:
                    temp.append(j)
                for k in topic2wordnum[:,word2index[word]]:
                    temp.append(k)
            else:
                for j in range(el+tl):
                    temp.append(0.0)
            c0 = c0 + 1
            record.append(temp)
            if(c0 == words_from_head):
                break
        for i in range(words_from_head-c0):
            temp = []
            for j in range(el+tl):
                temp.append(0.0)
            record.append(temp)

### For body
        c0 = 0
        for word in body:
            temp = []
            if(word in embedding_dict):

                for j in embedding_dict[word]:
                    temp.append(j)
                for k in topic2wordnum[:,word2index[word]]:
                    temp.append(k)
            else:
                for j in range(el+tl):
                    temp.append(0.0)
            c0 = c0 + 1
            record.append(temp)
            if(c0 == words_from_body):
                break
        for i in range(words_from_body-c0):
            temp = []
            for j in range(el+tl):
                temp.append(0.0)
            record.append(temp)

        final_feature.append(record)

    final_feature_tensor = torch.tensor(np.array(final_feature))
    final_target = torch.tensor(inp['Stance'].values)
    return (final_feature_tensor,final_target)
def create_embeddings(model,newser,inp_train,inp_test,words_from_head,words_from_body):
     print("creating embeddings")
     # imp_dic,ed = get_word_pos()
     # doc2topic,topic2word = generate_matrices(model,newser.values,imp_dic)
     # pickle.dump(topic2word,open("topic2word.pk" , "wb"))
     # pickle.dump(ed,open("embedding_dict.pk" , "wb"))
     # pickle.dump(imp_dic,open("word2index.pk" , "wb"))
     ed = pickle.load(open("embedding_dict.pk" , "rb"))
     imp_dic = pickle.load(open("word2index.pk" , "rb"))
     topic2word = pickle.load(open("topic2word.pk" , "rb"))
     doc2topic = topic2word
     return (topic2word,doc2topic,ed,imp_dic)
     # print("shape of topic2word is ")
     # print(topic2word.shape)
     # X_train,y_train = parse_df_method_concat(inp_train,words_from_head,words_from_body,ed,topic2word,imp_dic)
     # X_test,y_test = parse_df_method_concat(inp_test,words_from_head,words_from_body,ed,topic2word,imp_dic)
     # pickle.dump(X_train,open("training_inp","wb"))
     # pickle.dump(y_train,open("training_target","wb"))
     # pickle.dump(X_test,open("testing_inp","wb"))
     # pickle.dump(y_test,open("testing_target","wb"))
     # print("shape of training set is ")
     # print(X_train.shape)
     # print("shape of testing set is ")
     # print(X_test.shape)
     # print("creation of embeddings over")
