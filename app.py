import pickle
import pandas as pd
from tensorflow.keras.models import load_model
from utils import Converter, modelTrain, predictor, pdf2text
import warnings

warnings.filterwarnings("ignore")


def main():
    df = pd.read_csv("job_titles_skills .csv")
    skills = df["job_skills"].str.lower()
    position = df["job_title"].str.lower()
    unique_skills = df["job_skills"].unique().tolist()
    comp_text = ""
    skill_index = []
    vocab_size_in = 10000
    vocab_size_out = 3000

    skills_conv = Converter(skills, vocab_size_in)
    skills_conv.sequences()

    pos_conv = Converter(position, vocab_size_out, "")
    pos_conv.sequences()

    y = pos_conv.toCat()

    model = modelTrain(
        vocab_size_in, skills_conv.max_len, vocab_size_out, skills_conv.pad_data, y
    )

    model.save("model_lstm_git.h5")
    with open("index_to_word_git.pkl", "wb") as f:
        pickle.dump(pos_conv.index_to_word, f)

    with open("token_skills_git.pkl", "wb") as f:
        pickle.dump(skills_conv.tok, f)

    with open("index_to_word_real.pkl", "rb") as file1:
        index_to_word = pickle.load(file1)

    with open("token_skills_git_real.pkl", "rb") as file2:
        token_skills = pickle.load(file2)

    model = load_model("lstm_model_real.h5")

    data = pdf2text()

    for i in data:
        comp_text += f"{i} "
        for j in unique_skills:
            if i in j.split():
                skill_index.append(i)
                break

    result = predictor(
        token_skills, model, comp_text, skills_conv.max_len, pos_conv.word_index
    )

    print(result)


if __name__ == "__main__":
    main()
