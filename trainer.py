import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
import gc
from sklearn.metrics import accuracy_score
def train(train_optimizer,train_criterion,train_device,train_model,target,totalDocs,docsPerFile,epochs,trainString,batch_size):
    print("in training")
    temp_input = []
    temp_target = []
    losses = []
    training_matrix = []
    for epoch in range(epochs):
        print("epoch num is " + str(epoch))
        h0,c0 = train_model.init_hidden()
        h0 = h0.to(train_device)
        c0 = c0.to(train_device)
        for iterator in range((totalDocs+docsPerFile-1)//docsPerFile):

            del training_matrix
            gc.collect()
            starting = iterator*docsPerFile
            ending = min((iterator+1)*docsPerFile-1,totalDocs-1)
            print(str(starting) + " " + str(ending))
            training_matrix = pickle.load(open(trainString + "^" + str(starting) + "^" + str(ending)+".pk" , "rb"))
            for batch_num in range(docsPerFile// batch_size):
                print("in batch number %d" % (batch_num))
                if(batch_size*(batch_num+1)  -1 >= training_matrix.shape[0] ):
                    print("not enough samples in batch")
                    break
                del temp_input
                gc.collect()
                temp_input = torch.tensor(training_matrix[batch_num*batch_size:batch_size*(batch_num+1),:,:]).to(train_device)
                temp_target= target[batch_num*batch_size:batch_size*(batch_num+1)].to(train_device)
                temp_input = temp_input.to(torch.float32)
                train_optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    out, hidden = train_model(temp_input, (h0, c0))
                    loss = train_criterion(out, temp_target)
                    loss.backward()
                    train_optimizer.step()
        losses.append(loss.item())
    return losses


def test(train_optimizer,train_criterion,train_device,train_model,target,totalDocs,docsPerFile,epochs,trainString,batch_size):
    print("in testing")
    temp_input = []
    temp_target = []
    batch_acc = []
    training_matrix = []
    for epoch in range(epochs):
        print("epoch num is " + str(epoch))
        h0,c0 = train_model.init_hidden()
        h0 = h0.to(train_device)
        c0 = c0.to(train_device)
        for iterator in range((totalDocs+docsPerFile-1)//docsPerFile):

            del training_matrix
            gc.collect()
            starting = iterator*docsPerFile
            ending = min((iterator+1)*docsPerFile-1,totalDocs-1)
            print(str(starting) + " " + str(ending))
            training_matrix = pickle.load(open(trainString + "^" + str(starting) + "^" + str(ending)+".pk" , "rb"))
            for batch_num in range(docsPerFile// batch_size):
                print("in batch number %d" % (batch_num))
                if(batch_size*(batch_num+1) -1 >= training_matrix.shape[0] ):
                    print("not enough samples in batch")
                    break
                del temp_input
                gc.collect()
                temp_input = torch.tensor(training_matrix[batch_num*batch_size:batch_size*(batch_num+1),:,:]).to(train_device)
                temp_target= target[batch_num*batch_size:batch_size*(batch_num+1)].to(train_device)
                temp_input = temp_input.to(torch.float32)
                train_optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    out, hidden = train_model(temp_input, (h0, c0))
                    _, preds = torch.max(out, 1)
                    preds = preds.to("cpu").tolist()
                    batch_acc.append(accuracy_score(preds, temp_target.tolist()))
    return (sum(batch_acc)/len(batch_acc))
