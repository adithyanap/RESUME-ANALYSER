import pickle
import pandas as pd

from utils import Converter, modelTrain, predictor, pdf2text

df = pd.read_csv("job_skills.csv")
skills = df["job_skills"].str.lower()
position = df["job_title"].str.lower()


vocab_size_in = 10000
vocab_size_out = 3000

skills_conv = Converter(skills, vocab_size_in)
skills_conv.sequences()


pos_conv = Converter(position, vocab_size_out)
pos_conv.sequences()

y = pos_conv.toCat()


model = modelTrain(
    vocab_size_in, skills_conv.max_len, vocab_size_out, skills_conv.pad_data, y
)

index_to_word = {}
for word, index in pos_conv.word_index.items():
    index_to_word[index] = word

model.save("model_lstm_git.h5")
with open("index_to_word_git.pkl", "wb") as f:
    pickle.dump(index_to_word, f)

with open("token_skills_git.pkl", "wb") as f:
    pickle.dump(skills_conv.tok, f)


with open("index_to_word.pkl", "rb") as file1:
    index_to_word = pickle.load(file1)


with open("token_skills.pkl", "rb") as file2:
    token_skills = pickle.load(file2)
