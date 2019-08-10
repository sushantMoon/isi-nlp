import math
import argparse
from collections import Counter
from nltk.corpus import brown

START = '<S>'
'''Start symbol---two are prepended to the start of every sentence.'''

STOP = '</S>'
'''Stop symbol---one is appended to the end of every sentence.'''

UNKNOWN = '<UNK>'
'''Unknown word symbol used to describe any word that is out of vocabulary.'''

UNKNOWN_THRESHOLD = 2
'''If the count of a token is less than this number, it should be treated as
out of vocabulary.'''

TRAIN = 'data/brown.txt'

class ngram_language_model:
    def __init__(self, *args, **kwargs):
        self.unknown = UNKNOWN
        self.unknown_threshold = UNKNOWN_THRESHOLD
        self.first_word = START
        self.last_word = STOP
        self.training_file = TRAIN
        # model (tuple of Counter): (unigrams, bigrams, trigrams) maps from
        # n-grams to counts
        # n_gram_probabilities (tuple of dict): (unigrams_to_probs, bigrams_to_probs,
        # trigrams_to_probs) dicts that cache the probabilities when they
        # are calculated
        self.model = ()
        self.n_gram_probabilities = ()
        self.unknown_words = None
        self.training_phase = True


    def download_brown_corpus(self):
        with open(self.training_file, 'w') as f:
            for l in brown.sents():
                line = ' '.join(l)
                f.write(line+'\n')


    def add_n_gram_counts(self, n, n_grams, tokens):
        """Adds the n-grams to the specified Counter from the specified tokens."""
        for i in range(len(tokens) - (n - 1)):
            n_grams[tuple(tokens[i:i+n])] += 1
        return n_grams


    def train(self):
        """
        Trains an n-gram model using the specified corpus.
        Args:
            training_file (str): the training_file of the corpus
        Returns:
            A tuple of (Counter, Counter, Counter) which are Counter objects from
            (tuple, int) that are the counts of the unigrams, bigrams, and
            trigrams respectively.
        """
        unigrams = Counter()
        bigrams = Counter()
        trigrams = Counter()

        lines = list()
        with open(self.training_file) as f:
            for line in f:
                lines.append(line)

        # creating the uni-gram counts
        for line in lines:
            tokens = line.split()
            tokens.insert(0, self.first_word)
            tokens.insert(0, self.first_word)
            tokens.append(self.last_word)
            self.add_n_gram_counts(1, unigrams, tokens)

        # the set of all uni-grams that have a count less than unknown_threshold
        unks = set()
        num_unks = 0
        for unigram, count in unigrams.items():
            if count < self.unknown_threshold:
                unks.add(unigram[0])
                num_unks += count

        for word in unks:
            del unigrams[(word,)]

        unigrams[(self.unknown,)] = num_unks
        self.unknown_words = unks
        # creating the bigram and trigram counts
        for line in lines:
            tokens = [token if token not in self.unknown_words else self.unknown for token in line.split()]
            tokens.insert(0, self.first_word)
            tokens.insert(0, self.first_word)
            tokens.append(self.last_word)
            self.add_n_gram_counts(2, bigrams, tokens)
            self.add_n_gram_counts(3, trigrams, tokens)

        return unigrams, bigrams, trigrams
    

    def train_log_prob(self, n_gram):
        """
        This is a nested method that will be passed to eval_model().
        Args:
            n_gram (tuple of str): represents an n-gram
        Returns:
            The log probability of the specified n-gram under the model.
        """

        # unigram
        if len(n_gram) == 1:
            if n_gram in self.n_gram_probabilities[0]:
                log_prob = self.n_gram_probabilities[0][n_gram]
            else:
                uni_numer = self.model[0][n_gram]
                uni_denom = sum(self.model[0].values()) - self.model[0][(self.first_word,)]
                log_prob = math.log(uni_numer / uni_denom, 2)
                self.n_gram_probabilities[0][n_gram] = log_prob

        # bigram
        if len(n_gram) == 2:
            if n_gram in self.n_gram_probabilities[1]:
                log_prob = self.n_gram_probabilities[1][n_gram]
            else:
                bi_numer = self.model[1][n_gram]
                bi_denom = self.model[0][n_gram[:1]]
                if bi_numer == 0:
                    return float('-inf')
                log_prob = math.log(bi_numer / bi_denom, 2)
                self.n_gram_probabilities[1][n_gram] = log_prob

        # trigram
        if len(n_gram) == 3:
            if n_gram in self.n_gram_probabilities[2]:
                log_prob = self.n_gram_probabilities[2][n_gram]
            else:
                tri_numer = self.model[2][n_gram]
                tri_denom = self.model[1][n_gram[:2]]
                if tri_denom == 0:
                    vocab_size = len(self.model[0]) - 1
                    return math.log(1 / vocab_size, 2)
                if tri_numer == 0:
                    return float('-inf')
                log_prob = math.log(tri_numer / tri_denom, 2)
                self.n_gram_probabilities[2][n_gram] = log_prob

        return log_prob


    def get_log_probablities(self, n_gram):
        log_prob = float('-inf')

        # unigram
        if len(n_gram) == 1:
            if n_gram in self.n_gram_probabilities[0]:
                log_prob = self.n_gram_probabilities[0][n_gram]

        # bigram
        if len(n_gram) == 2:
            if n_gram in self.n_gram_probabilities[1]:
                log_prob = self.n_gram_probabilities[1][n_gram]

        # trigram
        if len(n_gram) == 3:
            if n_gram in self.n_gram_probabilities[2]:
                log_prob = self.n_gram_probabilities[2][n_gram]

        return log_prob


    def main(self):
        print('Training n-gram model on', self.training_file, '...')
        print('Minimum count for a word to be known:', self.unknown_threshold)

        unigrams, bigrams, trigrams = self.train()
        self.model = (unigrams, bigrams, trigrams)

        print('vocab size is', len(unigrams) - 1)
        print('num tokens is', sum(self.model[0].values()) - self.model[0][(self.first_word,)])
        print('num unknown words is', self.model[0][(self.unknown,)])
        print()

        # (unigrams_to_probs, bigrams_to_probs, trigrams_to__probs)
        self.n_gram_probabilities = (dict(), dict(), dict())

        for n in range(1, 4):
            print('evaluating', str(n) + '-gram model ...')
            print('evaluating on', self.training_file,'train set ...')
            perplexity = self.eval_model(n)
            print('perplexity:', str(perplexity))
            print()


    def eval_model(self, n):
        """Returns the perplexity of the model on a specified test set."""

        log_prob_sum = 0
        file_word_count = 0

        with open(self.training_file) as f:
            for line in f:
                prob, num_tokens = self.eval_sentence(n, line)
                log_prob_sum += prob
                file_word_count += num_tokens
            f.close()

        average_log_prob = log_prob_sum / file_word_count
        perplexity = 2**(-average_log_prob)
        return perplexity


    def eval_sentence(self, n, sentence):
        """
        Returns log probability of a sentence and how many tokens were in the
            sentence.
        """
        tokens = []
        for token in sentence.split():
            if (token,) in self.model[0]:
                tokens.append(token)
            else:
                tokens.append(self.unknown)
        # tokens = [token if (token,) in self.model[0] else self.unknown for token in sentence.split()]
        num_tokens = len(tokens) + 1
        for _ in range(1, n):
            tokens.insert(0, self.first_word)
        tokens.append(self.last_word)

        log_prob_sum = 0
        for i in range(len(tokens) - (n - 1)):
            n_gram = tuple(tokens[i:i+n])
            next_prob = self.train_log_prob(n_gram) if self.training_phase else self.get_log_probablities(n_gram)
            log_prob_sum += next_prob

        return log_prob_sum, num_tokens


    def make_predictions(self, msg, number_of_predictions, n_gram_model):
        """
        makes prediction for the next possible words using the available words
        """
        msg = ' '.join([x for x in msg.strip().split()])
        predictions = []
        for (word, ) in self.model[0]:
            predicted_sentence = msg + ' ' + word
            total_log_prob, num_tokens = self.eval_sentence(n_gram_model, predicted_sentence)
            predictions.append((predicted_sentence, total_log_prob))
        
        predictions.sort( key=lambda x: x[1], reverse=True)
        print(predictions[:number_of_predictions])


if __name__ == '__main__':
    ng = ngram_language_model()
    ng.download_brown_corpus()
    ng.main()
    ng.training_phase = False
    ng.make_predictions("I am ", 5, 1)