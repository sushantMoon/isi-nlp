# NLP : Query Prediction

## Assignment 1

### N-gram Model

`n_gram.py` : This script creates and tests the N-grams basically it works using trigrams.

It calculates the frequencies of the words, tuple and trigrams occuring together in training phase and in the testing phase calculates the probability of the differnet words occuring together with the given query and makes the predictions

### LSTM

#### Training

`lstm.py` : This script is used for training

##### Note

1. Pay attention to Batch size used for trianing.
2. Training is done on Brown Corpus `nltk.corpus.brown`

##### Deployment/Testing

`query_prediction_lstm.py`: This script is used for making the predictions by the flask server.

### Running

In the shell runn : `python server.py`
In the browser open : `localhost:8081`

## References

1. LayoutIt.com
2. [Generate Song Lyrics using LSTM](https://medium.com/coinmonks/word-level-lstm-text-generator-creating-automatic-song-lyrics-with-neural-networks-b8a1617104fb)
