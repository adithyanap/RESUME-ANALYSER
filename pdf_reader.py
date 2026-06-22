import pypdf
import string
import sys
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class FormatError(Exception):
    pass


name, ex = sys.argv[1].split(".")

if ex != "pdf":
    raise FormatError("file format error")
nltk.download("stopwords")
nltk.download("punkt")


def pdf2text(path):
    file = pypdf.PdfReader(path)
    print("test")
    data = []
    full_text = ""
    word = ""
    temp_text = ""
    sus = [",", ".", "(", ")", "/", ":", "@", "&"]
    for i in file.pages:
        text = i.extract_text().lower()
        text += full_text
    for j in range(len(text)):
        if text[j] in sus:
            temp_text += "\n"
        else:
            temp_text += text[j]

    for p in temp_text:
        if p != " ":
            word += p
        if p == "\n":
            data.append(word[:-1])
            word = ""

    return data


data = pdf2text(sys.argv[1])
print(data)
