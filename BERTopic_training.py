from bertopic import BERTopic
import numpy as np
import pandas as pd
import pickle
def train_and_create(input,words_per_topic=10,nr_topics_1 = None):
    print("training model")
    # mod = BERTopic(calculate_probabilities  = True , top_n_words = words_per_topic,nr_topics = nr_topics_1)
    # mod.fit(input)
    mod = pickle.load(open('trained_bert.pk' , 'rb'))
    print("model training over")
    return mod
