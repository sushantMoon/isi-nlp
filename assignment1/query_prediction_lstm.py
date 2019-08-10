import nltk
import numpy as np
from keras.models import load_model


class lstm:
    def __init__(self):
        self.MIN_SEQUENCE_LEN = 4
        self.model = load_model('checkpoints/final-lstm_model.h5')

        self.dictionary = list()
        self.word_indices = dict()
        self.indices_word = dict()
        with open('data/dictionary') as file:
            data = file.readlines()
            for line in data:
                word, index = line.split()
                self.dictionary.append(word)
                self.word_indices[word] = int(index)
                self.indices_word[int(index)] = word
        return None
    
    def preprocess_input(self, message):
        x = np.zeros((1, self.MIN_SEQUENCE_LEN-1, len(self.dictionary)), dtype=np.bool)
        if len(message) > 2:
            message = message[-3:]
            for t, w in enumerate(message):
                if w in self.word_indices:
                    x[0, t, self.word_indices[w]] = 1
                else:
                    pass
        elif len(message) == 2:
            for t, w in enumerate(message):
                if w in self.word_indices:
                    x[0, t+1, self.word_indices[w]] = 1
                else:
                    pass
        elif len(message) == 1:
            for t, w in enumerate(message):
                if w in self.word_indices:
                    x[0, 2, self.word_indices[w]] = 1
                else:
                    pass
        return x
    
    def sample(self, preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def get_prediction(self, message, quantity=10):
        message = [x.strip() for x in message.strip().split()]
        if len(message) < 1:
            return {}
        x = self.preprocess_input(message)
        y = self.model.predict(x)
        
        probs = y[0]
        predictions = {}
        for _ in range(quantity):
            next_index = self.sample(probs)
            next_word = self.indices_word[next_index]
            probability = y[0][self.dictionary.index(next_word)]
            predictions[next_word] = str(probability)
            np.delete(probs, [next_index])
        return predictions

if __name__ == "__main__":
    # from query_prediction_lstm import lstm
    # lstm = lstm()
    # lstm.make_prediction("This is the result of")
    pass