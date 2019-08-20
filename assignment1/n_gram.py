import nltk
from nltk.corpus import brown
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline


class ngram_language_model:
    def __init__(self):
        self.model = None
        self.valid_text = None

    def build_and_train_model(self):
        text = brown.sents(
            categories=[
                'adventure', 'belles_lettres', 'editorial', 'fiction',
                'government', 'hobbies', 'humor', 'learned', 'lore',
                'mystery', 'news', 'religion', 'reviews', 'romance',
                'science_fiction'
            ]
        )
        valid_text = []
        for sentence in text:
            words = []
            for word in sentence:
                words.extend(nltk.word_tokenize(word))
            valid_text.append(words)

        self.valid_text = valid_text

        n = 3       # length of largest everygram

        train_data, padded_sents = padded_everygram_pipeline(n, valid_text)

        self.model = MLE(n)
        self.model.fit(train_data, padded_sents)
        return

    def make_predictions(self, msg, number_of_predictions=5):
        """
        makes prediction for the next possible words using the available words
        """
        sentence = []
        for x in msg.strip().split():
            sentence.extend(nltk.word_tokenize(x))
        alpha = 0.1
        beta = 0.3
        gamma = 0.6
        predictions = []
        prediction_dict = {}
        for word in self.model.vocab:
            alpha_prob = alpha*self.model.score(word)
            beta_prob = beta*self.model.score(word, sentence[-1:])
            gamma_prob = gamma*self.model.score(word, sentence[-2:])
            prob = alpha_prob + beta_prob + gamma_prob
            predictions.append((word, prob))
        predictions.sort(key=lambda x: x[1], reverse=True)
        for word, prob in predictions[:number_of_predictions]:
            prediction_dict[word] = prob
        return prediction_dict


if __name__ == '__main__':
    # ng = ngram_language_model()
    # ng.build_and_train_model()
    # ng.make_predictions("I am ")
    pass
