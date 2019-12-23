import argparse


def program():
    argument_parser = argparse.ArgumentParser(description='Simple word2vec implementation')
    # input, output
    argument_parser.add_argument('--corpusFile', type=str, help='Path to file with corpus', required=True)
    argument_parser.add_argument('--outputFile', type=str, help='Path to file with output embeddings')
    # skip gram
    argument_parser.add_argument('--windowSize', type=int, help='Size of context words near target word', default=2)
    argument_parser.add_argument('--negativeSamples', type=int,
                                 help='Value of negative samples for each target word', default=4)
    argument_parser.add_argument('--vectorDim', type=int, help='Size of word vector', default=100)
    # training
    argument_parser.add_argument('--epochs', type=int, help='Number of training epochs', default=500)
    argument_parser.add_argument('--batchSize', type=int, help='Size of training batch')
    args = argument_parser.parse_args()


if __name__ == "__main__":
    program()
