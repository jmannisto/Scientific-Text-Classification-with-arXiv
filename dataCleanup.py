import pandas as pd
import json
from nltk.corpus import stopwords #note: make sure to have nltk downloaded

#if we want to use LabelEncoder for converting our categories
#from sklearn.preprocessing import LabelEncoder 

#Read the data
data = pd.read_json('arxivSmall.json')
#NOTE: we can also use arxiv API by using arxiv on github: https://github.com/lukasschwab/arxiv.py

#lowercase
data['abstract'] = data['abstract'].str.lower()

#TODO: Convert categories
# data['abstract'] = data['abstract'].replace([what], [with what?])
#can also use LabelEncoder

#remove numbers and punctuation
data['abstract'] = data['abstract'].str.replace(r'[^\w\s]+', '', regex=True)

#TODO: remove stop words 
#QUESTION: should this be done before or after the punctuation? 
stop = stopwords.words('english')
test['abstract'] = test['abstract'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

#TODO: split into test and train data
