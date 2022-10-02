import pandas as pd
import json
from nltk.corpus import stopwords #note: make sure to have nltk downloaded
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#if we want to use LabelEncoder for converting our categories
#from sklearn.preprocessing import LabelEncoder 

#Read the data
data = pd.read_json('arxivSmall.json')
#NOTE: we can also use arxiv API by using arxiv on github: https://github.com/lukasschwab/arxiv.py

#lowercase - easier to do before tokenizing
data['abstract'] = data['abstract'].str.lower()

#remove numbers and punctuation - easier to do before tokenizing 
data['abstract'] = data['abstract'].str.replace(r'[^\w\s]+', '', regex=True)
#removes any token that is non-alphanumeric or begins with a number

#do we need to sentence tokenize first?
#data['abstract'] = data['abstract'].apply(sent_tokenize)

#Tokenize
data['abstract'] = data['abstract'].apply(word_tokenize)

#remove stop words  
stop = stopwords.words('english')
data['abstract'] = data['abstract'].apply(lambda x: [token for token in x if token not in stop])

#remove category subidentifies (indicated by a period after the category)
#remove any instance of '.' followed by at least two letters or . followed by at least two letters, a dash '-' and more letters
data["categories"] = data["categories"].str.replace(r"\.[A-Za-z]{2,}\b|\.[A-Za-z]{2,}-[A-Za-z]{1,}\b","", regex=True) 

#remove duplicate categories
data["categories"] = data["categories"].str.replace(r'\b([\w-]+)( \1\b)+', r'\1', regex=True)


#convert categories into arrays
data["categories"] = data["categories"].str.split(' ') 

#TODO: Encode categories (run into ValueError: setting an array element with a sequence if not)

#TODO: tfidf
#defaults at first
#tweak pending results 
vectorizer = TfidfVectorizer(max_df = 1.0)
abstracts = data.abstract.astype(str) #can't pass tokens through vectorizer 
tfidf_matrix = vectorizer.fit_transform(abstracts)
features = vectorizer.get_feature_names_out()

#TODO: normalize # of vectors within each row

#split into test and train data
X = data['abstract']
y = data['categories']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#TODO: Train model
neigh = KNeighborsClassifier(n_neighbors=5) #play around with K
neigh.fit(X_train, y_train)
