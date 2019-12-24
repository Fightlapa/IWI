import os
import numpy as np
import pandas as pd


def read_input_file(path, word_separator=" "):
    if os.path.isdir(path):
        raise NameError('Path is folder')

    if not os.path.isfile(path):
        raise NameError('Path is not a file')

    if str.split(path, ".")[1] != 'txt':
        raise NameError('Not a txt file')

    word_sequence = []
    with open(path) as file:
        for line in file:
            for word in line.split(word_separator):
                word_sequence.append(word.strip())

    return word_sequence


def create_vocabulary(word_sequence):
    word_dict = dict()
    # extract unique words in corpus
    for word in word_sequence:
        word_dict[word] = True

    # assign index values to vocabulary
    word_to_index = dict()
    index_to_word = dict()
    for index, word in enumerate(sorted(word_dict)):
        word_to_index[word] = index
        index_to_word[index] = word

    return word_to_index, index_to_word


def transform_to_index_array(word_sequence, word_to_index):
    transformed_sequence = [word_to_index[word] for word in word_sequence]
    indexed_sequence_array = np.asarray(transformed_sequence, dtype=np.int)
    return indexed_sequence_array


def write_output_embeddings(path, index_to_word, target_layer):
    np_index = np.empty(shape=len(index_to_word), dtype=object)
    for index, word in index_to_word.items():
        np_index[index] = word
    df = pd.DataFrame(data=target_layer, index=np_index)
    df.to_csv(path, float_format="%.4f", header=False)
