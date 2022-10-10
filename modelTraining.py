from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load

#Read the data
df = pd.read_json('cleanedData.json')

#pulling out top 20 categories
data = df[df["categories"].isin(["math", "cs", "hep-ph", "astro-ph", "physics-ph", "quant-ph",
                                                   "cond-mat", "hep-th", "q-fin", "hep-lat", "math-ph", "econ",
                                                   "stat", "q-bio", "nucl-th", "nlin", "eess", "nucl-ex",
                                                   "gr-qc", "hep-ex"])]

#pulling samples
sampleData = data[data["categories"] == "math"].sample(n = 20000)
sampleData = data.append(data[data["categories"] == "cs"].sample(n = 20000))
sampleData = sampleData.append(data[data["categories"] == "hep-ph"].sample(n = 20000))
sampleData = sampleData.append(data[data["categories"] == "astro-ph"].sample(n = 20000))
sampleData = sampleData.append(data[data["categories"] == "physics-ph"].sample(n = 20000))
sampleData = sampleData.append(data[data["categories"] == "quant-ph"].sample(n = 20000))
sampleData = sampleData.append(data[data["categories"] == "cond-mat"].sample(n = 20000))
sampleData = sampleData.append(data[data["categories"] == "hep-th"].sample(n = 20000))
sampleData = sampleData.append(data[data["categories"] == "q-fin"].sample(n = 20000))
sampleData = sampleData.append(data[data["categories"] == "hep-lat"].sample(n = 20000, replace = True))
#sampleData = sampleData.append(data[data["categories"] == "math-ph"].sample(n = 20000, replace = True)) #error?
sampleData = sampleData.append(data[data["categories"] == "econ"].sample(n = 20000, replace = True))
sampleData = sampleData.append(data[data["categories"] == "stat"].sample(n = 20000))
sampleData = sampleData.append(data[data["categories"] == "q-bio"].sample(n = 20000))
sampleData = sampleData.append(data[data["categories"] == "nucl-th"].sample(n = 20000))
sampleData = sampleData.append(data[data["categories"] == "nlin"].sample(n = 20000, replace = True))
sampleData = sampleData.append(data[data["categories"] == "eess"].sample(n = 20000, replace = True))
sampleData = sampleData.append(data[data["categories"] == "nucl-ex"].sample(n = 20000))
sampleData = sampleData.append(data[data["categories"] == "gr-qc"].sample(n = 20000))
sampleData = sampleData.append(data[data["categories"] == "hep-ex"].sample(n = 20000))

#splitting test and train data ahead of time
X = sampleData["lemma abstract"]
y = sampleData['encoded_categories']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


#TF IDF with lemmas (performs better than stems)
vectorizer = TfidfVectorizer(max_df = 0.75)
X_train_knn = vectorizer.fit_transform(X_train)
X_test_knn = vectorizer.transform(X_test)

#KNN Classifier 
#results = 0.7724
knn = KNeighborsClassifier(n_neighbors=21) 
knn.fit(X_train_knn, y_train)
print("KNN Classifier Accuracy)
print(knn.score(X_test_knn, y_test)) 
      
#model persistence
dump(knn, 'knn_model.joblib') 
#load with:
#knn = load('knn_model.joblib') 

#dummy classifier for comparison
#results = 0.104
from sklearn.dummy import DummyClassifier
dummy_classifier = DummyClassifier(strategy = "most_frequent")
dummy_classifier.fit(X_train_knn, y_train)
print("Dummy Predictor Accuracy:")
print(dummy_classifier.score(X_test_knn, y_test)) 
