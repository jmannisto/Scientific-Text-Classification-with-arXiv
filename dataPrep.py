import pandas as pd
from nltk.corpus import stopwords #note: make sure to have nltk downloaded
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

#fileName = "cleanedData.json"
#to upload category dictionary:
def loadCategoryDict(file_name):
    with open(file_name) as json_file:
        categoryKeysLoaded = json.load(json_file)

def createTrainingData(sample_size, data):
    sampleSize = sample_size
    #creates two dataframes, one that only includes data that has one category assignment, another that has data that has two category assignments
    oneCat,twoCat = dfSplit(data)
    #balance data
    balancedData = getSamples(sampleSize, oneCat, twoCat)
    X_train, X_test, y_train, y_test = splitTestTrain(balancedData)
    model_train, model_test, vectorizer = tfidfVectorize(X_train, X_test)
    return model_Train, model_test, y_train, y_test, vectorizer

def cleanData(fileName):
    data = pd.read_json(fileName)
    data = lowercase(data)
    data = removePunct(data)
    data = removeStop(data)
    data = lemmatize(data)
    #remove abstracts that have less than 50 words
    data = removeShort(data)
    data["categories"] = removeCategory(data)
    data["categories"] = removeDuplicates(data)
    data = encode(data)
    data = categoryCheck(data)
    return data

#lowercase abstracts
def lowercase(data):
    data['abstract'] = data['abstract'].str.lower()
    return data

#removes any token that is non-alphanumeric or begins with a number
def removePunct(data):
    data['abstract'] = data['abstract'].str.replace(r'[^\w\s]+', '', regex=True)
    return(data)

#remove stop words
def removeStop(data):
    stop = stopwords.words('english')
    data["abstract"] = data['abstract'].apply(
        lambda x: ' '.join([word for word in str(x).split() if word not in (stop)]))
    return data

#lemmatize and add new column to dataframe with lemmatized abstract
def lemmatize(data):
    wnl = WordNetLemmatizer()
    data["lemma abstract"] = data["abstract"].apply(lambda x: ' '.join([wnl.lemmatize(word) for word in str(x).split()]))
    return data

# Remove articles with abstracts less than 50 words
def removeShort(data):
    length = []
    [length.append(len(str(text))) for text in data['abstract']]
    data['length'] = length
    data = data.drop(data['abstract'][data['length'] < 50].index, axis=0)
    return data

#remove subcategories of larger categories
def removeCategory(data):
  #remove any instance of '.' followed by at least two letters or . followed by at least two letters, a dash '-' and more letters
    data["categories"] = data["categories"].str.replace(r"\.[A-Za-z]{2,}\b|\.[A-Za-z]{2,}-[A-Za-z]{1,}\b","", regex=True)
    #combine all categories for physics and cond-mat into one
    data["categories"] =  data["categories"].str.replace(r"\bphysics.*\b", "physics", regex=True)
    data["categories"] =  data["categories"].str.replace(r"\bcond-mat.*\b", "cond-mat", regex=True)
    return data["categories"]

#remove duplicate categories
def removeDuplicates(data):
    data["categories"] = data["categories"].str.replace(r'\b([\w-]+)( \1\b)+', r'\1', regex=True)
    return data["categories"]

#Encode categories
def encode(data):
    le = LabelEncoder()
    le.fit(data['categories'])
    data['encoded_categories'] = le.transform(data['categories'])
    return data

#removes data entries that have more than two categories
def categoryCheck(data):
    data = data[data['categories'].str.split().str.len().lt(3)]
    return data

#split into two dataframes:
def dfSplit(data):
    #create dataframe that only has one category
    soloData = data[data['categories'].str.split().str.len().lt(2)] #only 1 category
    #create dataframe that has more than one but less than 3 categories (i.e. 2)
    duoData = data[data['categories'].str.split().str.len().lt(3)]
    duoData = data[data['categories'].str.split().str.len().gt(1)]
    return soloData, duoData

#balance the data and get samples
def getSamples(sample_size, soloData, duoData):
    balancedData = pd.DataFrame()
    catGroup = soloData.groupby("categories").count().reset_index()
    for category in catGroup.categories:
        try:
            #try sampling the data
            balancedData = balancedData.append(soloData[soloData["categories"] == category].sample(n = sample_size))
        
        except ValueError: #thrown when data size is too small for sample
            #calculate how much is needed from double cat group to reach sample-size
            amtNeed = (sample_size - (catGroup.loc[catGroup["categories"] == category, 'lemma abstract'].item()))
            #find relevant double category groups that have the category we're search for in them
            contain_values = duoData[duoData['categories'].str.contains(category)]
            #sample the needed sample size amount from double category group
            contain_values_sampled = contain_values.sample(n=amtNeed, replace=True)  # oversample if needed
            #combine datasets 
            sampleData = pd.concat([contain_values_sampled, soloData[soloData['categories'].str.contains(category)]])
            #add sample data to the collection of balanced data
            balancedData = balancedData.append(sampleData)
    return balancedData

def splitTestTrain(sampleData):
    # splitting test and train data ahead of time
    X = sampleData["lemma abstract"]
    y = sampleData['encoded_categories']
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return train_test_split(X, y, test_size=0.2)

#TF IDF with lemmas (performs better than stems)
def tfidfVectorize(X_train, X_test):
    vectorizer = TfidfVectorizer(max_df = 0.75)
    X_train_matrix = vectorizer.fit_transform(X_train)
    X_test_matrix = vectorizer.transform(X_test)
    return X_train_matrix, X_test_matrix, vectorizer

#split into two dataframes:
def dfSplit(data):
    #create dataframe that only has one category
    soloData = data[data['categories'].str.split().str.len().lt(2)] #only 1 category
    #create dataframe that has more than one but less than 3 categories (i.e. 2)
    duoData = data[data['categories'].str.split().str.len().lt(3)]
    duoData = data[data['categories'].str.split().str.len().gt(1)]
    return soloData, duoData

#balance the data and get samples
def getSamples(sample_size, soloData, duoData):
    balancedData = pd.DataFrame()
    catGroup = soloData.groupby("categories").count().reset_index()
    for category in catGroup.categories:
        try:
            #try sampling the data
            balancedData = balancedData.append(soloData[soloData["categories"] == category].sample(n = sample_size))
        
        except ValueError: #thrown when data size is too small for sample
            #calculate how much is needed from double cat group to reach sample-size
            amtNeed = (sample_size - (catGroup.loc[catGroup["categories"] == category, 'lemma abstract'].item()))
            #find relevant double category groups that have the category we're search for in them
            contain_values = duoData[duoData['categories'].str.contains(category)]
            #sample the needed sample size amount from double category group
            contain_values_sampled = contain_values.sample(n=amtNeed, replace=True)  # oversample if needed
            #combine datasets 
            sampleData = pd.concat([contain_values_sampled, soloData[soloData['categories'].str.contains(category)]])
            #add sample data to the collection of balanced data
            balancedData = balancedData.append(sampleData)
    return balancedData

def splitTestTrain(sampleData):
    # splitting test and train data ahead of time
    X = sampleData["lemma abstract"]
    y = sampleData['encoded_categories']
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return train_test_split(X, y, test_size=0.2)

#TF IDF with lemmas (performs better than stems)
def tfidfVectorize(X_train, X_test):
    vectorizer = TfidfVectorizer(max_df = 0.75)
    X_train_matrix = vectorizer.fit_transform(X_train)
    X_test_matrix = vectorizer.transform(X_test)
    return X_train_matrix, X_test_matrix, vectorizer
