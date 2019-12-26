import argparse

import pandas as pd
from scipy.spatial.distance import cdist


def predict():
    parser = argparse.ArgumentParser(description='A word2vec implementation in Keras')
    parser.add_argument('--embeddings', type=str, help='Path to output embeddings', required=True)
    args = parser.parse_args()

    log_format = '%(asctime)-15s %(message)s'
    embeddings = pd.read_csv(args.embeddings, index_col=0)

    while True:
        word = input("""Please enter a word (enter "EXIT" to quit): """)
        if word == "EXIT":
            return
        elif word not in embeddings.index:
            print("Could not find word '%s' in index" % word)
        else:
            query = embeddings.loc[word]
            scores = 1.0 - cdist(query.to_frame().T, embeddings, 'cosine')
            embeddings['score'] = scores[0]
            print(embeddings['score'].sort_values(ascending=False)[0:10])


if __name__ == "__main__":
    predict()