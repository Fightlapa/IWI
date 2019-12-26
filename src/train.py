import argparse
import src.data_processing as data_processing
from src.model import TrainingModel

# TODO
# Tworzenie datasetu
# Trening

def program():
    argument_parser = argparse.ArgumentParser(description='Simple word2vec implementation')
    # input, output
    argument_parser.add_argument('--corpusFile', type=str, help='Path to txt file with corpus', required=True)
    argument_parser.add_argument('--outputFile', type=str, help='Path to file with output embeddings')
    # skip gram
    argument_parser.add_argument('--windowSize', type=int, help='Size of context words near target word', default=2)
    argument_parser.add_argument('--negativeSamples', type=int,
                                 help='Value of negative samples for each target word', default=4)
    argument_parser.add_argument('--vectorDim', type=int, help='Size of word vector', default=100)
    # training
    argument_parser.add_argument('--epochs', type=int, help='Number of training epochs', default=500)
    argument_parser.add_argument('--batchSize', type=int, help='Size of training batch', default=4)
    args = argument_parser.parse_args()

    # process input data
    word_sequence = data_processing.read_input_file(args.corpusFile)
    word_to_index, index_to_word = data_processing.create_vocabulary(word_sequence)
    vocabulary_size = len(word_to_index)
    indexed_sequence_array = data_processing.transform_to_index_array(word_sequence, word_to_index)

    # train model
    training_model = TrainingModel(vocabulary_size, args.vectorDim)
    training_model.build_model()
    training_model.train_model(indexed_sequence_array, args.windowSize, args.batchSize, args.negativeSamples, args.epochs)

    # write trained embeddings
    if args.outputFile:
        data_processing.write_output_embeddings(args.outputFile, index_to_word,
                                                training_model.model.get_layer("target_layer").get_weights()[0])


if __name__ == "__main__":
    program()
