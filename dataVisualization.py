import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image
import seaborn as sns

#visualization of # of articles by category
#plot top 20 categories
highToLow = data.groupby('categories').size().sort_values(ascending=False)
highToLow.iloc[:20].plot(kind='bar')

#plotting categories that have at least 10,000 articles
ax = highToLow.iloc[:20].plot(kind='barh', title = 'Categories with at least 10,000 articles')
ax.bar_label(ax.containers[0])
plt.xlabel("Article Count")
plt.ylabel("Categories")

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

#List out highest tfidf scores:
score = tfidf_matrix[0]
df = pd.DataFrame(score.T.todense(), index=vectorizer.get_feature_names_out(), columns=["Tfidf"]) 
df.sort_values(by=["Tfidf"],ascending=False)
df.sort_values(by=["Tfidf"],ascending=False).head(10)

K = 15 # choose K highest
print("Words with highest TF/IDF:")
# get indices of words with highest TF/IDF score 
#(np.flip for descending order of indices)
idx = np.flip(np.argsort(tfidf_matrix[0, :].A)[0][-K:])
scores = np.flip(np.sort(tfidf_matrix[0, :].A)[0][-K:])
words = vectorizer.get_feature_names_out()[idx]

print(dict(zip(words, scores)))
