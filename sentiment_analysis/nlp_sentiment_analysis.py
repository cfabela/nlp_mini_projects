import collections
import nltk
import os
from sklearn import (
    datasets, feature_extraction
)


def extract_features(corpus):
    count_vectorizer = feature_extraction.text.CountVectorizer(
        lowercase=True,
        tokenizer=nltk.word_tokenize,
        stop_words='english',
        min_df=1 # minimum document frequency, the word must appear more than once
    )
    processed_corpus = count_vectorizer.fit_transform(corpus)
    processed_corpus = feature_extraction.text.TfidfTransformer().fit_transform(
        processed_corpus)
    return processed_corpus


data_directory = 'movie_reviews'
movie_sentiment_data = datasets.load_files(data_directory,shuffle=True)
print('{} files loaded.'.format(len(movie_sentiment_data.data)))
print('They contain the following classes: {}.'.format(movie_sentiment_data.target_names))

movie_tfidf = extract_features(movie_sentiment_data.data)
print(movie_tfidf)