#Exercise 4 NLP

import nltk
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from pathlib import Path
from string import digits, punctuation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering

#2 Speeches 1
#2a) reading text file into corpus

speechfiles = sorted(Path('data/speeches').glob('[R][0]*'))
#print(speechfiles)

corpus = []
for filenames in speechfiles:
    try:
        f = open(filenames, 'r')
        corpus.append(f.readlines()[0])
        f.close()
    except UnicodeDecodeError:
        print(filenames)

#2b) vectorizing speeches using 1,2,3-grams
_stemmer = nltk.snowball.SnowballStemmer('english')

def tokenize_and_stem(text):
    """Return tokens of text deprived of numbers and interpunctuation."""
    text = text.translate(str.maketrans({p: "" for p in digits + punctuation}))
    return [_stemmer.stem(t) for t in nltk.word_tokenize(text.lower())]

_stopwords = nltk.corpus.stopwords.words('english')
_stopstring = " ".join(_stopwords)
_stopwords = tokenize_and_stem(_stopstring)

tfidf = TfidfVectorizer(stop_words=_stopwords, ngram_range=(1, 3), tokenizer=tokenize_and_stem)

tfidf_matrix = tfidf.fit_transform(corpus)

print(tfidf.vocabulary_)
tfidf_matrix.todense()

terms = tfidf.get_feature_names()

df = pd.DataFrame(tfidf_matrix.todense().T, index=terms)
columns = list(df.columns.values)
index=1
for column in columns:
    columns[index - 1] = "Value " + str(index)
    index +=1
values = df.values.tolist()
df_n = pd.DataFrame(values, columns=columns)
print(df_n)

#2c) pickling sparse matrix

count_matrix = open("output/speech_matrix.pk", "wb")
pickle.dump(tfidf_matrix, count_matrix)
count_matrix.close()

df_t = pd.DataFrame(terms)
df_t.to_csv(r'output/terms.csv')

#3
#3a) reading count matrix
count_matrix = open("output/speech_matrix.pk", "rb")
speeches = pickle.load(count_matrix)

#3b) creating dendogram

speeches_d = speeches.toarray()
clustering = AgglomerativeClustering().fit(speeches_d)

matrix = linkage(speeches_d, method="complete", metric="cosine")
plt.figure(figsize=(10, 10))
dendrogram(matrix)

plt.tick_params(
    axis='x',
    which='both',
    labelbottom=False)
plt.show()

#3c saving dendogram
plt.savefig('output/speeches_dendrogram.pdf')


#4 Job Ads
#4a) reading file, parsing lines, setting datetime type

df = pd.read_csv('data/Stellenanzeigen.txt')
list = []
with open("./data/Stellenanzeigen.txt", "r") as stellenanzeigen:
    for row in stellenanzeigen:
        row = row.strip('\n')
        if row != '':
            list.append(row)

_stemmer = nltk.snowball.SnowballStemmer('german')
def tokenize_and_stem(text):
    """Return tokens of text deprived of numbers and interpunctuation."""
    text = text.translate(str.maketrans({p: "" for p in digits + punctuation}))
    return [_stemmer.stem(t) for t in nltk.word_tokenize(text.lower())]

_stopwords = nltk.corpus.stopwords.words('german')
_stopstring = " ".join(_stopwords)
_stopwords = tokenize_and_stem(_stopstring)


# I give up on this exercise, sorry! Deleted the trial code
