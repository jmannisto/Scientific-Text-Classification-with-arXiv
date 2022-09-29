import pandas as pd
import json
from nltk.corpus import stopwords #note: make sure to have nltk downloaded
from nltk.tokenize import word_tokenize

#if we want to use LabelEncoder for converting our categories
#from sklearn.preprocessing import LabelEncoder 

#Read the data
data = pd.read_json('arxivSmall.json')
#NOTE: we can also use arxiv API by using arxiv on github: https://github.com/lukasschwab/arxiv.py

#Tokenize
data['abstract'] = word_tokenize(data['abstract'])
#data['abstract'] = data['abstract'].apply(word_tokenize)

#lowercase
data['abstract'] = [token.str.lower() for token in data['abstract']]

#remove numbers and punctuation
data['abstract'] = [token for token in data['abstract'] if token != r'[\W(\d.*)]']
#removes any token that is non-alphanumeric or begins with a number

#remove stop words  
stop = stopwords.words('english')
data['abstract'] = [token for token in data['abstract'] if token not in stop]
#test['abstract'] = test['abstract'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

#remove category subidentifies (indicated by a period after the category)
data["categories"] = data["categories"].str.replace(r"\..+\b","", regex=True)
#remove duplicate categories
data["categories"] = data["categories"].str.replace(r'\b(\w+)( \1\b)+', r'\1', regex=True)

#TODO: Convert categories
#convert categories into arrays
#
#hold off on labeling numerically, change if needed during training process
#data['categories'] = data['categories'].replace([what], [with what?])
#can also use LabelEncoder

#TODO: split into test and train data

