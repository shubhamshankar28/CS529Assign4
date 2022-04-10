import numpy as np
import pandas as pd
import  pickle
def parse( words_required,input_string,topic2word,embedding_dict,word2index):
    docSpecificMatrix = []
    word_counter = 0
    first_word = list(embedding_dict.items())[0]
    # print(first_word)
    # print(words_required)

    length_of_matrix = topic2word.shape[0] + len(embedding_dict[first_word[0]])
    for words in input_string.split():
        # print(word_counter,words_required)
        if(word_counter == words_required):
            break
        wordMatrix = []
        if(words in embedding_dict):
            for value in embedding_dict[words]:
                wordMatrix.append(value)
            for value in topic2word[:,word2index[words]]:
                wordMatrix.append(value)
        else:
            for value in range(length_of_matrix):
                wordMatrix.append(0.0)
        docSpecificMatrix.append(wordMatrix)
        word_counter= word_counter + 1
    while(word_counter < words_required):
        wordMatrix = []
        for value in range(length_of_matrix):
            wordMatrix.append(0.0)
        docSpecificMatrix.append(wordMatrix)
        word_counter = word_counter + 1
    return docSpecificMatrix


def generateInputMatrix(topic2word,embedding_dict,word2index,text_df,words_in_head,words_in_body,namofset, size = 5000):
    # print(type(text_df))
    # print(text_df.columns)
    # print(text_df.loc[0,'Headline'])
    total_documents = text_df.shape[0]
    counter = 0
    start_index = -1
    finish_index = -1
    i = 0
    print(words_in_head)
    print(words_in_body)
    print(text_df.shape)
    indices = list(text_df.index)
    while(i<total_documents):

        print("creating pickle file for document:%d" % (i))
        if(counter == 0):
            start_index = i
            inputMatrix = []

            ph = parse(words_in_head,text_df.loc[indices[i],'Headline'],topic2word,embedding_dict,word2index)
            pb = parse(words_in_body,text_df.loc[indices[i],'articleBody'],topic2word,embedding_dict,word2index)
            p_fin = ph+pb
            inputMatrix.append(p_fin)
            # print(len(docSpecificMatrix))
            # print(len(docSpecificMatrix[0]))

            counter = counter+1
            i=i+1

        elif (counter == size):
            # print("dimensions are %d %d" % (len(inputMatrix) , len(inputMatrix.shape[0])))
            finish_index = i-1
            print("dumping begins")
            arr = np.array(inputMatrix)
            print(arr.shape)

            pickle.dump(arr,open(namofset+"^"+str(start_index)+"^"+str(finish_index)+".pk","wb"))
            print("dumping ends")
            counter = 0


        else:
            docSpecificMatrix = []
            ph = parse(words_in_head,text_df.loc[indices[i],'Headline'],topic2word,embedding_dict,word2index)
            pb = parse(words_in_body,text_df.loc[indices[i],'articleBody'],topic2word,embedding_dict,word2index)
            p_fin = ph+pb
            inputMatrix.append(p_fin)

            counter = counter+1
            i=i+1
    if(counter != 0):
        finish_index = total_documents - 1
        print("dumping begins")
        arr = np.array(inputMatrix)
        print(arr.shape)

        pickle.dump(arr,open(namofset+"^"+str(start_index)+"^"+str(finish_index)+".pk","wb"))
        print("dumping ends")
