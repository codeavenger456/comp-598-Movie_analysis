import json
import pandas as pd
import re
from math import log
import sys

PUNCTUATIONS = {'(', ')', '[', ']', ',', '-', '.', '?', '!', ':', ';', '#', '&'}
VALID = 3

def main():
    tweets = pd.read_csv("../data/tweets.csv", header=0, sep=",")
    tweets.dropna(subset=["Topic", "Sentiment"], inplace=True)
    tweets = tweets.astype(str)
    tweets = clean_tweet(PUNCTUATIONS, tweets)
    stop = get_stop_word()
    tweets = clean_column(tweets)
    tweets = remove_stop_alpha(stop, tweets)
    freq = build_word_freq_pony(tweets)
    result = tf_idf(freq, 10)
    with open("tfidf.json", "w") as f:
        f.write(json.dumps(result, indent=4))

def tf_idf(data, n_words):
    output = dict()
    for i in data.keys():
        output[i] = list()
        for j in data[i].keys():
            output[i].append((j, tf_idf_calc(data, i, j)))
        output[i].sort(key=lambda x: x[1], reverse=True)
        output[i] = output[i][:n_words]
        output[i] = [j[0] for j in output[i]]
    return output

def tf_idf_calc(data, topic, word):
    return data[topic][word] * idf(data, word)

def idf(data, word):
    count = 0
    for i in data.keys():
        if word in data[i].keys():
            count += 1
    return log(len(data.keys()) / count)

def build_word_freq_pony(data, valid=VALID):
    pre_output = dict()
    output = dict()
    for i in pd.unique(data["Topic"]):
        pre_output[i] = dict()
        output[i] = dict()
    word_freq = data["Tweet"].str.split().apply(pd.value_counts).fillna(0)
    word_freq = word_freq[word_freq.columns[word_freq.sum() >= VALID]]
    topics = data["Topic"].tolist()
    for (i, j), k in zip(word_freq.iterrows(), topics):
        for l in word_freq.columns:
            pre_output[k][l] = pre_output[k].get(l, 0) + j[l]
    for i in pre_output.keys():
        for (j, k) in pre_output[i].items():
            if k >= VALID:
                output[i][j] = k
    return output

def remove_stop_alpha(stop, dialog):
    data = dialog.copy()
    process = data["Tweet"].tolist()
    for i in range(len(process)):
        process[i] = " ".join([j for j in process[i].split() if j.isalpha() and j not in stop])
    data["Tweet"] = process
    return data

def clean_column(data):
    data["Topic"] = data["Topic"].str.strip()
    data["Sentiment"] = data["Sentiment"].str.strip()
    return data

def clean_tweet(punctuations, data):
    data["Tweet"] = data["Tweet"].str.lower()
    data["Tweet"] = [re.sub("@[A-Za-z0-9_]+","", str(i)) for i in data["Tweet"]]
    data["Tweet"] = [re.sub("#[A-Za-z0-9_]+","", str(i)) for i in data["Tweet"]]
    data["Tweet"] = [re.sub(r"http\S+", "", str(i)) for i in data["Tweet"]]
    data["Tweet"] = [re.sub(r"www.\S+", "", str(i)) for i in data["Tweet"]]
    for i in punctuations:
        data["Tweet"] = data["Tweet"].str.replace(i, ' ')
    return data


def get_stop_word():
    stop = set()
    with open("../data/stopwords.txt") as f:
        while True:
            line = f.readline()
            if not line.startswith('#'):
                break
        while line:
            stop.add(line.strip('\n'))
            line = f.readline()
    f.close()
    return stop

if __name__ == "__main__":
    main()