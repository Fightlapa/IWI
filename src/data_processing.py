import os
import re
import string

import numpy as np
import pandas as pd

def read_corpus_file(path, word_separator=" "):
    if os.path.isdir(path):
        raise NameError('Path is folder')

    if not os.path.isfile(path):
        raise NameError('Path is not a file')

    if str.split(path, ".")[1] != 'txt':
        raise NameError('Not a txt file')

    try:
        with open(path) as file:
            word_sequence = extract_words(file, word_separator)
    except Exception:
        with open(path, encoding="utf-8") as file:
            word_sequence = extract_words(file, word_separator)

    return word_sequence[1:]

def read_input_file(embeddings_path, context_path):
    if not embeddings_path or not context_path:
        return [], []

    vocabulary_emb, embedding_weights = get_weights(embeddings_path)
    vocabulary_con, context_weights = get_weights(context_path)
    assert vocabulary_emb == vocabulary_con

    return vocabulary_emb, embedding_weights, context_weights


def get_weights(path):
    if os.path.isdir(path):
        raise NameError('Path is folder')
    if not os.path.isfile(path):
        raise NameError('Path is not a file')
    if str.split(path, ".")[1] != 'csv':
        raise NameError('Not a txt file')
    vocabulary = []
    weights = []
    try:
        with open(path, encoding="utf-8") as file:
            for line in file:
                splitted_line = line.split(',')
                vocabulary.append(splitted_line[0])
                weights.append(splitted_line[1:])
    except Exception:
        with open(path) as file:
            for line in file:
                splitted_line = line.split(',')
                vocabulary.append(splitted_line[0])
                weights.append(splitted_line[1:])
    return vocabulary, weights


def extract_words(file, word_separator):
    word_sequence = []
    for line in file:
        for word in line.split(word_separator):
            word = word.strip()
            if word != '' and not any(i.isdigit() for i in word):
                word_sequence.append(word.lower())
    return word_sequence


def create_vocabulary(word_sequence, input_embeddings):
    word_dict = dict()
    input_dict = dict()
    word_unigram = list()
    # extract unique words in corpus
    for word in word_sequence:
        if not word.isalpha():
            word = re.sub(r'\W+', '', word)
        if word != '':
            word_dict[word] = True
            word_unigram.append(word)

    for word in input_embeddings:
        input_dict[word] = True

    # assign index values to vocabulary
    word_to_index = dict()
    index_to_word = dict()

    for index, word in enumerate(sorted(input_dict)):
        index_to_word[index] = word
        word_to_index[word] = index

    real_index = 0
    for _, word in enumerate(sorted(word_dict)):
        if word not in word_to_index.keys():
            real_index = real_index + len(input_dict)
            index_to_word[real_index] = word
            word_to_index[word] = real_index
            real_index += 1

    return word_to_index, index_to_word, word_unigram


def transform_to_index_array(word_sequence, word_to_index):
    transformed_sequence = []
    for word in word_sequence:
        word_only_letters = re.sub(r'\W+', '', word)
        if word_only_letters == '':
            continue
        if word.endswith('.'):
            word_only_letters = re.sub(r'\W+', '', word)
            transformed_sequence.append(word_to_index[word_only_letters])
            transformed_sequence.append(-1) # means it's dot ending sequence
        else:
            transformed_sequence.append(word_to_index[word_only_letters])
    indexed_sequence_array = np.asarray(transformed_sequence, dtype=np.int)
    return indexed_sequence_array


def write_output_embeddings(path, index_to_word, target_layer):
    np_index = np.empty(shape=len(index_to_word), dtype=object)
    for index, word in index_to_word.items():
        np_index[index] = word
    df = pd.DataFrame(data=target_layer, index=np_index)
    df.to_csv(path, float_format="%.4f", header=False)

