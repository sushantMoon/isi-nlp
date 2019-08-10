import nltk
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Bidirectional, LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import TensorBoard

import numpy as np

MIN_SEQUENCE_LEN = 4
BATCH_SIZE=20480

def generator(sentence_list, next_word_list, dictionary, word_indices, batch_size):
    index = 0
    while True:
        x = np.zeros((batch_size, MIN_SEQUENCE_LEN-1, len(dictionary)), dtype=np.bool)
        y = np.zeros((batch_size, len(dictionary)), dtype=np.bool)
        for i in range(batch_size):
            for t, w in enumerate(sentence_list[index]):
                x[i, t, word_indices[w]] = 1
            y[i, word_indices[next_word_list[index]]] = 1
            index = index+1
            if index == len(sentence_list):
                index = 0
        yield x ,y


text = nltk.corpus.brown
valid_text = []
# count = 0
for sentence in text.sents():
    if len(sentence) >= MIN_SEQUENCE_LEN:
        valid_text.append(sentence)
    #     count += 1
    # if count > 100:
    #     break


dictionary=set()
target_word=[]
feature_sentence=[]
for sentence in valid_text:
    for word in sentence:
        dictionary.add(word)
    ngrams = nltk.ngrams(sentence, MIN_SEQUENCE_LEN)
    for ngram in ngrams:
        feature_sentence.append(ngram[:-1])
        target_word.append(ngram[-1])


(
    train_feature_sentence,
    test_feature_sentence,
    train_target_word,
    test_target_word
) = train_test_split (
        feature_sentence,
        target_word,
        test_size=0.02,
        shuffle=True,
        random_state=42
    )

dictionary = sorted(dictionary)
word_indices = dict((c, i) for i, c in enumerate(dictionary))
indices_word = dict((i, c) for i, c in enumerate(dictionary))

import nltk
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Bidirectional, LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import TensorBoard

import numpy as np

MIN_SEQUENCE_LEN = 4
BATCH_SIZE=20480

def generator(sentence_list, next_word_list, dictionary, word_indices, batch_size):
    index = 0
    while True:
        x = np.zeros((batch_size, MIN_SEQUENCE_LEN-1, len(dictionary)), dtype=np.bool)
        y = np.zeros((batch_size, len(dictionary)), dtype=np.bool)
        for i in range(batch_size):
            for t, w in enumerate(sentence_list[index]):
                x[i, t, word_indices[w]] = 1
            y[i, word_indices[next_word_list[index]]] = 1
            index = index+1
            if index == len(sentence_list):
                index = 0
        yield x ,y

text = nltk.corpus.brown
valid_text = []
# count = 0
for sentence in text.sents():
    if len(sentence) >= MIN_SEQUENCE_LEN:
        valid_text.append(sentence)
    #     count += 1
    # if count > 100:
    #     break


dictionary=set()
target_word=[]
feature_sentence=[]
for sentence in valid_text:
    for word in sentence:
        dictionary.add(word)
    ngrams = nltk.ngrams(sentence, MIN_SEQUENCE_LEN)
    for ngram in ngrams:
        feature_sentence.append(ngram[:-1])
        target_word.append(ngram[-1])


(
    train_feature_sentence,
    test_feature_sentence,
    train_target_word,
    test_target_word
) = train_test_split (
        feature_sentence,
        target_word,
        test_size=0.02,
        shuffle=True,
        random_state=42
    )

dictionary = sorted(dictionary)
word_indices = dict((c, i) for i, c in enumerate(dictionary))
indices_word = dict((i, c) for i, c in enumerate(dictionary))

with open('./data/dictionary', 'w') as file:
    for key in word_indices:
        file.write("{} {}\n".format(key, word_indices[key]))

model = Sequential()
model.add(
        Bidirectional(
            LSTM(128), 
            input_shape=(
                MIN_SEQUENCE_LEN-1,
                len(dictionary)
            )
        )
 )
model.add(Dropout(0.3))
model.add(Dense(len(dictionary), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

file_path = "./checkpoints/query_prediction-epoch{epoch:03d}-words%d-sequence%d-loss{loss:.4f}-cat_acc{categorical_accuracy:.4f}-val_loss{val_loss:.4f}-val_cat_acc{val_categorical_accuracy:.4f}" % ( len(dictionary), MIN_SEQUENCE_LEN-1)
checkpoint = ModelCheckpoint(file_path, monitor='val_categorical_accuracy', verbose=1, save_best_only=True, period=2)
early_stopping = EarlyStopping(monitor='val_categorical_accuracy')
tensorboard = TensorBoard(log_dir='./logs', update_freq=1000)
# callbacks_list = [checkpoint, early_stopping, tensorboard]
callbacks_list = [checkpoint, tensorboard]

model.fit_generator(
        generator(train_feature_sentence, train_target_word, dictionary, word_indices, BATCH_SIZE),
        steps_per_epoch=int(len(train_feature_sentence)/BATCH_SIZE) + 1,
        epochs=20,
        callbacks=callbacks_list,
        validation_data=generator(test_feature_sentence, test_target_word, dictionary, word_indices, BATCH_SIZE),
        validation_steps=int(len(test_feature_sentence)/BATCH_SIZE) + 1
)

model.save('./checkpoints/final-lstm_model.h5')
