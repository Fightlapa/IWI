from keras.initializers import RandomNormal
from keras.layers import Input, Dense, Reshape, merge
from keras.layers.core import Flatten
from keras.layers.embeddings import Embedding
from keras.layers.merge import dot
from keras.models import Model
from keras.optimizers import TFOptimizer
import keras.preprocessing.sequence as sequence
import tensorflow as tf
import numpy as np

import random

class TrainingModel:
    def __init__(self, vocabulary_size, vector_dim, word_unigram, word_to_index):
        self.vocabulary_size = vocabulary_size
        self.vector_dim = vector_dim
        self.model = None
        self.word_unigram = word_unigram
        self.word_to_index = word_to_index

    def build_model(self, existing_embedding_weights, existing_context_weights):
        stddev = 1.0 / self.vector_dim
        initializer = RandomNormal(mean=0.0, stddev=stddev, seed=None)

        target_input = Input(shape=(1,), name="target_input")
        target_layer = Embedding(input_dim=self.vocabulary_size, output_dim=self.vector_dim, input_length=1,
                                 name="target_layer", embeddings_initializer=initializer)(target_input)

        context_input = Input(shape=(1,), name="context_input")
        context_layer = Embedding(input_dim=self.vocabulary_size, output_dim=self.vector_dim, input_length=1,
                                  name="context_layer", embeddings_initializer=initializer)(context_input)

        merged = dot([target_layer, context_layer], axes=2, normalize=False, name="dot")
        merged = Flatten()(merged)
        output = Dense(1, activation='sigmoid', name="output")(merged)

        optimizer = TFOptimizer(tf.train.AdagradOptimizer(0.1))
        model = Model(inputs=[target_input, context_input], outputs=output)
        model.compile(loss="binary_crossentropy", optimizer=optimizer)


        embedding_weights = model.get_layer("target_layer").get_weights()
        context_weights = model.get_layer("context_layer").get_weights()

        for i in range(0, len(existing_embedding_weights)):
            embedding_weights[0][i] = existing_embedding_weights[i]

        for i in range(0, len(existing_context_weights)):
            context_weights[0][i] = existing_context_weights[i]

        model.get_layer("target_layer").set_weights(embedding_weights)
        model.get_layer("context_layer").set_weights(context_weights)

        model.summary()
        self.model = model

    def train_model(self, sequence, window_size, batch_size, negative_samples, epochs):
        # in order to balance out more negative samples than positive
        negative_weight = 1.0 / negative_samples
        class_weight = {1: 1.0, 0: negative_weight}

        sequence_length = len(sequence)
        approx_steps_per_epoch = (sequence_length * (
                window_size * 2.0) + sequence_length * negative_samples) / batch_size

        seed = 1
        batch_iterator = self.create_batch_iterator(sequence, window_size, negative_samples, batch_size, seed)

        self.model.fit_generator(batch_iterator,
                                 steps_per_epoch=approx_steps_per_epoch,
                                 epochs=epochs,
                                 class_weight=class_weight,
                                 max_queue_size=100)

    def skip_gram_iterator(self, sequence, window_size, negative_samples, seed):
        """ An iterator which at each step returns a tuple of (word, context, label) """
        sequence = np.asarray(sequence, dtype=np.int)
        sequence_length = sequence.shape[0]
        epoch = 0
        i = 0
        while True:
            window_start = max(0, i - window_size)
            window_end = min(sequence_length, i + window_size + 1)

            # Now eliminate dots
            for j in range(window_start, i):
                if sequence[j] == -1:
                    window_start = j + 1

            for j in range(i, window_end):
                if sequence[j] == -1:
                    window_end = j - 1

            for j in range(window_start, window_end):
                if i != j:
                    yield (sequence[i], sequence[j], 1)

            unigram_size = len(self.word_unigram)
            for negative in range(negative_samples):
                random_float = random.random()
                j = int(random_float * unigram_size)
                yield (sequence[i], self.word_to_index[self.word_unigram[j]], 0)

            i += 1
            if i == sequence_length:
                epoch += 1
                i = 0
            while sequence[i] == -1:
                i += 1
                if i == sequence_length:
                    epoch += 1
                    i = 0

    def create_batch_iterator(self, sequence, window_size, negative_samples, batch_size, seed):
        """ An iterator which returns training instances in batches """
        iterator = self.skip_gram_iterator(sequence, window_size, negative_samples, seed)
        words = np.empty(shape=batch_size)  # , dtype=DTYPE)
        contexts = np.empty(shape=batch_size)  # , dtype=DTYPE)
        labels = np.empty(shape=batch_size)  # , dtype=DTYPE)

        while True:
            for i in range(batch_size):
                word, context, label = next(iterator)
                words[i] = word
                contexts[i] = context
                labels[i] = label
            yield ([words, contexts], labels)