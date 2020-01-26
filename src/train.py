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
    argument_parser.add_argument('--outputContextFile', type=str, help='Path to file with output context embeddings')

    # pretrained embeddings
    argument_parser.add_argument('--inputFile', type=str, help='Path to file with output embeddings.'
                                                               ' Both target and context input files must be provided')
    argument_parser.add_argument('--inputContextFile', type=str, help='Path to file with output context embeddings'
                                                                      ' Both target and context input files must'
                                                                      ' be provided')

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
    word_to_index, index_to_word, word_unigram = data_processing.create_vocabulary(word_sequence)
    vocabulary_size = len(word_to_index)
    indexed_sequence_array = data_processing.transform_to_index_array(word_sequence, word_to_index)

    # create model
    training_model = TrainingModel(vocabulary_size, args.vectorDim, word_unigram, word_to_index)
    training_model.build_model()

    # train model
    training_model.train_model(indexed_sequence_array, args.windowSize, args.batchSize, args.negativeSamples,
                               args.epochs)

    # write trained embeddings
    if args.outputFile:
        data_processing.write_output_embeddings(args.outputFile, index_to_word,
                                                training_model.model.get_layer("target_layer").get_weights()[0])

    if args.outputContextFile:
        data_processing.write_output_embeddings(args.outputContextFile, index_to_word,
                                                training_model.model.get_layer("context_layer").get_weights()[0])


if __name__ == "__main__":
    program()
