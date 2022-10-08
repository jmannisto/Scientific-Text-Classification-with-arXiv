#Read the data
data = pd.read_json('cleanedData.json')

#pulling out top 20 categories
smallerData = data[data["categories"].isin(["math", "cs", "hep-ph", "astro-ph", "physics-ph", "quant-ph",
                                            "cond-mat", "hep-th", "cond-mat-el", "cs math", "cond-mat-sci", "cond-mat-mech",
                                            "stat", "cond-mat-hall", "nucl-th", "math-ph math", "eess cs", "cs stat",
                                            "gr-qc", "hep-ex"])]

#pulling samples
#sorry this is so sloppy :|

sampleData = smallerData[smallerData["categories"] == "math"].sample(n = 10000)
sampleData = sampleData.append(smallerData[smallerData["categories"] == "cs"].sample(n = 10000))
sampleData = sampleData.append(smallerData[smallerData["categories"] == "hep-ph"].sample(n = 10000))
sampleData = sampleData.append(smallerData[smallerData["categories"] == "astro-ph"].sample(n = 10000))
sampleData = sampleData.append(smallerData[smallerData["categories"] == "physics-ph"].sample(n = 10000))
sampleData = sampleData.append(smallerData[smallerData["categories"] == "quant-ph"].sample(n = 10000))
sampleData = sampleData.append(smallerData[smallerData["categories"] == "cond-mat"].sample(n = 10000))
sampleData = sampleData.append(smallerData[smallerData["categories"] == "hep-th"].sample(n = 10000))
sampleData = sampleData.append(smallerData[smallerData["categories"] == "cond-mat-el"].sample(n = 10000))
sampleData = sampleData.append(smallerData[smallerData["categories"] == "cs math"].sample(n = 10000))
sampleData = sampleData.append(smallerData[smallerData["categories"] == "cond-mat-sci"].sample(n = 10000))
sampleData = sampleData.append(smallerData[smallerData["categories"] == "cond-mat-mech"].sample(n = 10000))
sampleData = sampleData.append(smallerData[smallerData["categories"] == "stat"].sample(n = 10000))
sampleData = sampleData.append(smallerData[smallerData["categories"] == "cond-mat-hall"].sample(n = 10000))
sampleData = sampleData.append(smallerData[smallerData["categories"] == "nucl-th"].sample(n = 10000))
sampleData = sampleData.append(smallerData[smallerData["categories"] == "math-ph math"].sample(n = 10000))
sampleData = sampleData.append(smallerData[smallerData["categories"] == "eess cs"].sample(n = 10000))
sampleData = sampleData.append(smallerData[smallerData["categories"] == "cs stat"].sample(n = 10000))
sampleData = sampleData.append(smallerData[smallerData["categories"] == "gr-qc"].sample(n = 10000))
sampleData = sampleData.append(smallerData[smallerData["categories"] == "hep-ex"].sample(n = 10000))

#splitting test and train data ahead of time
from sklearn.model_selection import train_test_split
X = sampleData["stemmed abstract"]
y = sampleData['encoded_categories']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


#TF IDF with stems
from sklearn.feature_extraction.text import TfidfVectorizer 
vectorizer = TfidfVectorizer(max_df = 0.8, min_df = .1)
X_train_knn = vectorizer.fit_transform(X_train)
X_test_knn = vectorizer.transform(X_test)

#KNN Classifier attempt
#results are as follows
#k = 5, tfidf max_df = 0.75 min_df = 0.1, 10,000 data samples of 20 categories
#model accuracy: 0.32269444444444445

# k = 3, tfidf max_df = 0.8 min_df = 0.1, 10,000 data samples of 20 categories
#model accuracy: 0.3026842105263158

# k = 3, tfidf max_df = 0.8 min_df = 0.1, 16,000 data samples of 20 categories
#model accuracy: 0.2985069444444444

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3, algorithm = "brute", n_jobs=-1) 
neigh.fit(X_train_knn, y_train)
print(neigh.score(X_test_knn, y_test)) 

#dummy classifier for comparison
#results = 0.104
from sklearn.dummy import DummyClassifier
dummy_classifier = DummyClassifier(strategy = "most_frequent")
dummy_classifier.fit(X_train_knn, y_train
dummy_predict = dummy_classifier.predict(X_test_knn)
print("Dummy Predictor Accuracy:")
print(dummy_classifier.score(X_test_knn, y_test)) 

#Test out Random Forest - WAY faster than KNN at predicting
#Accuracy: 0.38680555555555557
    #TFIDF: max_df = .75, min_df = 0.1
    # samples of 10,000 from top 20 categories
from sklearn.ensemble import RandomForestClassifier                     
rfc = RandomForestClassifier(max_depth=15) # a lot slower if you don't set a max_depth... I didn't bother waiting
rfc.fit(X_train_knn, y_train)  
print(rfc.score(X_test_knn, y_test)) 
                             
#Test out SGD Classifier - WAY faster than KNN at predicting
#Accuracy: 0.31283333333333335
    #TFIDF: max_df = .75, min_df = 0.1
    # samples of 10,000 from top 20 categories
from sklearn.linear_model import SGDClassifier

sgd= SGDClassifier()
sgd.fit(X_train_knn, y_train)
#mean of accuracy
print(sgd.score(X_test_knn, y_test))
