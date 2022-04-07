#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
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
from BERTopic_training import train_and_create
from gen_embeddings import create_embeddings
import  pickle


class FNC_BERTopicModel(torch.nn.Module) :
    def __init__(self,  embedding_dim,words_in_head,words_in_body, hidden_dim=100, mlp_layers=100) :
        super().__init__()

        self.wh = words_in_head
        self.wb = words_in_body

        # The LSTM layer takes in the the embedding size and the hidden vector size.
        # The hidden dimension is up to you to decide, but common values are 32, 64, 128
        self.lstm_head = nn.LSTM(embedding_dim, hidden_dim,batch_first= True,bidirectional = True)
        self.lstm_body = nn.LSTM(embedding_dim,hidden_dim,batch_first = True,bidirectional = True)
        # We use dropout before the final layer to improve with regularization


        # The fully-connected layer takes in the hidden dim of the LSTM and
        #  outputs a a 3x1 vector of the class scores.
        self.fc_init = nn.Linear(6*hidden_dim, mlp_layers)
        self.rel = nn.ReLU()
        self.fc_final = nn.Linear(mlp_layers,2)


    def forward(self, x, hidden):
        """
        The forward method takes in the input and the previous hidden state
        """

        # The input is transformed to embeddings by passing it to the embedding layer
        outA,hiddenA = self.lstm_head(x[:,:self.wh,:])
        outB,hiddenB  = self.lstm_body(x[:,self.wh:,:])

        # The embedded inputs are fed to the LSTM alongside the previous hidden state
        hA = outA.sum(1)
        hB = outB.sum(1)
        ### calculate_comps
        compA = hA - hB
        compB = torch.cat((hA,hB) , 1)
        ##compC = torch.from_numpy(np.array([torch.dot(hA,hB)]))



        fincomp = torch.cat((compB,compA) , 1)

        ### feeding it to first_mlp
        out_ua = self.fc_init(fincomp)
        out_a = self.rel(out_ua)
        out_fin = self.fc_final(out_a)

        return out_fin, hidden

    def init_hidden(self):
        return (torch.zeros(1,batch_size,32), torch.zeros(1,batch_size,32))


# In[28]:


class FNC_BERTopicModel_withDot(torch.nn.Module) :
    def __init__(self,  embedding_dim,words_in_head,words_in_body, hidden_dim=100, mlp_layers=100) :
        super().__init__()

        self.wh = words_in_head
        self.wb = words_in_body

        # The LSTM layer takes in the the embedding size and the hidden vector size.
        # The hidden dimension is up to you to decide, but common values are 32, 64, 128
        self.lstm_head = nn.LSTM(embedding_dim, hidden_dim,batch_first= True,bidirectional = True)
        self.lstm_body = nn.LSTM(embedding_dim,hidden_dim,batch_first = True, bidirectional = True)
        # We use dropout before the final layer to improve with regularization


        # The fully-connected layer takes in the hidden dim of the LSTM and
        #  outputs a a 3x1 vector of the class scores.
        self.fc_init = nn.Linear(6*hidden_dim+1, mlp_layers)
        self.rel = nn.ReLU()
        self.fc_final = nn.Linear(mlp_layers,2)


    def forward(self, x, hidden):
        """
        The forward method takes in the input and the previous hidden state
        """

        # The input is transformed to embeddings by passing it to the embedding layer
        outA,hiddenA = self.lstm_head(x[:,:self.wh,:])
        outB,hiddenB  = self.lstm_body(x[:,self.wh:,:])

        # The embedded inputs are fed to the LSTM alongside the previous hidden state
        hA = outA.sum(1)
        hB = outB.sum(1)
        ### calculate_comps
        compA = hA - hB
        compB = torch.cat((hA,hB) , 1)
        ##compC = torch.from_numpy(np.array([torch.dot(hA,hB)]))
        dot_products = []
        for i in range(hA.shape[0]):
            hAmod= torch.sqrt(torch.dot(hA[i,:],hA[i,:]))
            hBmod = torch.sqrt(torch.dot(hB[i,:],hB[i,:]))
            dot_products.append([torch.dot(hA[i,:],hB[i,:])/((hAmod+1)*(hBmod+1))])
        compC = torch.tensor(dot_products,dtype = torch.float32)



        fincomp = torch.cat((compB,compA,compC) , 1)

        ### feeding it to first_mlp
        out_ua = self.fc_init(fincomp)
        out_a = self.rel(out_ua)
        out_fin = self.fc_final(out_a)

        return out_fin, hidden

    def init_hidden(self):
        return (torch.zeros(1,batch_size,32), torch.zeros(1,batch_size,32))


# In[34]:


def trainer(train_model,train_optimizer,train_criterion,train_device):
    epochs = 40
    losses = []
    for e in range(epochs):

        h0, c0 =  train_model.init_hidden()

        h0 = h0.to(train_device)
        c0 = c0.to(train_device)

    #     for i in range(800):

    #         input = X_train[i,:,:]
    #         input = input.to(torch.float32)
    #         target = y_train[i]

    #         optimizer.zero_grad()
    #         with torch.set_grad_enabled(True):
    #             out, hidden = model(input, (h0, c0))
    #             loss = criterion(out, target)
    #             loss.backward()
    #             optimizer.step()
        for batch_idx, batch in enumerate(train_dl):

            input = batch[0].to(train_device)
            target = batch[1].to(train_device)
            input = input.to(torch.float32)

            train_optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                out, hidden = train_model(input, (h0, c0))
                loss = train_criterion(out, target)
                loss.backward()
                train_optimizer.step()
        losses.append(loss.item())
    return losses


# In[35]:


def tester(test_model,test_optimizer,test_device):
    batch_acc = []
    for batch_idx, batch in enumerate(test_dl):
        h0, c0 =  test_model.init_hidden()
        input = batch[0].to(test_device)
        target = batch[1].to(test_device)
        input = input.to(torch.float32)
        test_optimizer.zero_grad()
        with torch.set_grad_enabled(False):
            out, hidden = test_model(input, (h0, c0))
            _, preds = torch.max(out, 1)
            preds = preds.to("cpu").tolist()
            batch_acc.append(accuracy_score(preds, target.tolist()))

    return (sum(batch_acc)/len(batch_acc))


# In[41]:


def train_and_test(model,model_name,device):
    model.to(device)
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 3e-4)
    print("training begins")
    losses = trainer(model,optimizer,criterion,device)
    print("training ends")
    print("losses are ")
    ###print(losses)

    finacc = tester(model,optimizer,device)
    print("for " + model_name + "accuracy is ")
    print(finacc)


mode = int(sys.argv[1])
if(mode == 0):
    print("using primary mode of retrieval")
    data_set = pd.read_csv('FNC_Bin_Train.csv')
    full_set= data_set[:1000]
    newser = full_set['articleBody'] + full_set['Headline']
    train_set,test_set = train_test_split(full_set,test_size = 0.2)
    model = train_and_create(newser)
    X_train,y_train,X_test,y_test = create_embeddings(model,newser,train_set,test_set,12,300)
else:
    print("using alternate mode of retrieval")
    X_train = pickle.load(open("training_inp","rb" ))
    y_train = pickle.load(open("training_target" , "rb"))
    X_test = pickle.load(open("testing_inp" , "rb"))
    y_test = pickle.load(open("testing_target" , "rb"))



batch_size = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_ds = TensorDataset(X_train,y_train)
test_ds = TensorDataset(X_test,y_test)
train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, drop_last=True)
test_dl = DataLoader(test_ds, shuffle=True, batch_size=batch_size, drop_last=True)


# In[32]:


model = FNC_BERTopicModel(X_test.shape[2],12,300,70,15)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 3e-4)


train_and_test(FNC_BERTopicModel(X_test.shape[2],12,300,70,15) , "FNC_Bert_no_dot" , device)

# In[33]:

#plt.plot(losses)



# In[36]:


# model_dot = FNC_BERTopicModel_withDot(X_test.shape[2],12,300,70,15)
# model_dot.to(device)
# print(model_dot)
# criterion_dot = nn.CrossEntropyLoss()
# optimizer_dot = torch.optim.Adam(model_dot.parameters(), lr = 3e-4)
#

# In[37]:


# losses = trainer(model_dot,optimizer_dot,criterion_dot,device)



# In[38]:

#
# finacc = tester(model_dot,optimizer_dot,device)
# print(finacc)


# In[17]:




# In[42]:
