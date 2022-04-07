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
def train_and_create(input,words_per_topic=10,nr_topics_1 = None):
    print("training model")
    mod = BERTopic(calculate_probabilities  = True , top_n_words = words_per_topic,nr_topics = nr_topics_1)
    mod.fit(input)
    pickle.dump(mod,open('trained_bert.pk','wb'))
    print("model training over")
    return mod
