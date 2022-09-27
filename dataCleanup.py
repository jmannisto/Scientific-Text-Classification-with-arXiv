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

#lowercase
data['abstract'] = [token.str.lower() for token in data['abstract']]

#TODO: Convert categories
# data['abstract'] = data['abstract'].replace([what], [with what?])
#can also use LabelEncoder

#remove numbers and punctuation
data['abstract'] = [token for token in data['abstract'] if token != r'[\W(\d.*)]']
#removes any token that is non-alphanumeric or begins with a number

#TODO: remove stop words 
#QUESTION: should this be done before or after the punctuation? 
stop = stopwords.words('english')
data['abstract'] = [token for token in data['abstract'] if token not in stop]
#test['abstract'] = test['abstract'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

#TODO: split into test and train data
