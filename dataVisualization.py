import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image

#TODO: visualization of test/train data by category

# Wordcloud of high use words across categories
K = 20 # choose some more words for the wordclouds
idx = np.flip(np.argsort(tfidf_matrix[0, :].A)[0][-K:])
scores = np.flip(np.sort(tfidf_matrix[0, :].A)[0][-K:])
words = vectorizer.get_feature_names_out()[idx]

wordcloud = WordCloud(background_color="white", width=1000, height=1000, scale=4)
wordcloud.generate_from_frequencies(dict(zip(words, scores)))

plt.figure(figsize=(5, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.title("TF/IDF Wordcloud")
plt.axis("off")
plt.show()


#TODO: wordcloud of high use words within categories

#TODO: plot the tfidf scores of words?

#List out highest tfidf scores:
K = 15 # choose K highest

print("Words with highest TF/IDF:")

# get indices of words with highest TF/IDF score 
#(np.flip for descending order of indices)
idx = np.flip(np.argsort(tfidf_matrix[0, :].A)[0][-K:])
scores = np.flip(np.sort(tfidf_matrix[0, :].A)[0][-K:])
words = vectorizer.get_feature_names_out()[idx]

print(dict(zip(words, scores)))
