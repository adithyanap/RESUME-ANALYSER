import pandas as pd
import numpy as np
import string
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pypdf
import sys


class Converter:
    def __init__(self, data, vocab_size):
        self.data = data
        self.vocab_size = vocab_size

    def sequences(self):
        self.tok = Tokenizer(num_words=self.vocab_size, oov_token="<OO>")

        self.tok.fit_on_texts(self.data)

        self.seq = self.tok.texts_to_sequences(self.data)

        self.max_len = max(len(seq) for seq in self.seq)

        self.pad_data = pad_sequences(self.seq, padding="pre", maxlen=self.max_len)

        self.word_index = self.tok.word_index

        self.indices = np.array([seq[-1] for seq in self.seq])

    def toCat(self):

        return to_categorical(self.indices, num_classes=self.vocab_size)


def modelTrain(vocab_size_in, max_len, vocab_size_out, X, y):
    lstm_model = Sequential()
    lstm_model.add(
        Embedding(
            input_dim=vocab_size_in,
            output_dim=100,
            input_length=max_len,
            mask_zero=True,
        )
    )
    lstm_model.add(
        LSTM(
            units=128,
            return_sequences=True,
        )
    )
    lstm_model.add(Dropout(0.2))
    lstm_model.add(LSTM(units=128))
    lstm_model.add(Dense(units=vocab_size_out, activation="softmax"))
    lstm_model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    early_stop = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    history_lstm = lstm_model.fit(
        X, y, epochs=60, batch_size=32, validation_split=0.2, callbacks=[early_stop]
    )
    return lstm_model


def predictor(tok_skills, lstm_model, text, max_len, word_index_pos):
    index_to_word = {}
    for word, index in word_index_pos.items():
        index_to_word[index] = word
    text = text.lower()
    seq = tok_skills.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=max_len, padding="pre")
    pre_index = np.argmax(lstm_model.predict(seq))
    print(pre_index)
    return index_to_word[pre_index]


def pdf2text():
    class FormatError(Exception):
        pass

    if len(sys.argv) < 2:
        sys.exit(1)

    file_path = sys.argv[1]

    if not file_path.lower().endswith(".pdf"):
        raise FormatError("File format error: Input must be a PDF file.")

    file = pypdf.PdfReader(file_path)
    print("test")
    data = []
    full_text = ""
    word = ""
    temp_text = ""
    sus = [",", ".", "(", ")", "/", ":", "@", "&"]
    for i in file.pages:
        full_text += i.extract_text().lower()
    for j in range(len(full_text)):
        if full_text[j] in sus:
            temp_text += "\n"
        else:
            temp_text += full_text[j]

    for p in temp_text:
        if p != " ":
            word += p
        if p == "\n":
            data.append(word[:-1])
            word = ""

    return data
