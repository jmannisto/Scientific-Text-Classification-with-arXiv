import pandas as pd
import json
from nltk.corpus import stopwords #note: make sure to have nltk downloaded
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder 

#Read the data
data = pd.read_json('arxivSmall.json')

#lowercase
data['abstract'] = data['abstract'].str.lower()

#remove numbers and punctuation
data['abstract'] = data['abstract'].str.replace(r'[^\w\s]+', '', regex=True)
#removes words that is non-alphanumeric or begins with a number

#remove stop words  
stop = stopwords.words('english')
data["abstract"] = data['abstract'].apply(lambda x: ' '.join([word for word in str(x).split() if word not in (stop)]))

#lemmatize and add new column to dataframe with lemmatized abstract
wnl = WordNetLemmatizer()
data["lemma abstract"] = data["abstract"].apply(lambda x: ' '.join([wnl.lemmatize(word) for word in str(x).split()]))

# Remove articles with abstracts less than 50 words
length = []
[length.append(len(str(text))) for text in data['abstract']]
data['length'] = length
data = data.drop(data['abstract'][data['length'] < 50].index, axis=0)

#remove category subidentifies (indicated by a period after the category)
data["categories"] = data["categories"].str.replace(r"\.[A-Za-z]{2,}\b|\.[A-Za-z]{2,}-[A-Za-z]{1,}\b","", regex=True)

#remove duplicate categories
data["categories"] = data["categories"].str.replace(r'\b([\w-]+)( \1\b)+', r'\1', regex=True)

#encode categories
le = preprocessing.LabelEncoder()
le.fit(data['categories'])
data['encoded_categories'] = le.transform(data['categories'])

