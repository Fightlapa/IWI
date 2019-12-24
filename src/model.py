from keras.initializers import RandomNormal
from keras.layers import Input, Dense
from keras.layers.core import Flatten
from keras.layers.embeddings import Embedding
from keras.layers.merge import dot
from keras.models import Model
from keras.optimizers import TFOptimizer
import tensorflow as tf


class TrainingModel:
    def __init__(self):
        self.model = None

    def build_model(self, vocabulary_size, vector_dim):
        stddev = 1.0 / vector_dim
        initializer = RandomNormal(mean=0.0, stddev=stddev, seed=None)

        target_input = Input(shape=(1,), name="target_input")
        target_layer = Embedding(input_dim=vocabulary_size, output_dim=vector_dim, input_length=1,
                                 name="target_layer", embeddings_initializer=initializer)(target_input)

        context_input = Input(shape=(1,), name="context_input")
        context_layer = Embedding(input_dim=vocabulary_size, output_dim=vector_dim, input_length=1,
                                  name="context_layer", embeddings_initializer=initializer)(context_input)

        merged = dot([target_layer, context_layer], axes=2, normalize=False, name="dot")
        merged = Flatten()(merged)
        output = Dense(1, activation='sigmoid', name="output")(merged)

        optimizer = TFOptimizer(tf.train.AdagradOptimizer(0.1))
        model = Model(inputs=[target_input, context_input], outputs=output)
        model.compile(loss="binary_crossentropy", optimizer=optimizer)
        model.summary()
        self.model = model

    # TODO
    def train_model(self):
        return
