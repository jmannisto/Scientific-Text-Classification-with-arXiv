class Vectorizer:
    def __init__(self, categories):

        self.categories = categories
        self.vectorizer = CountVectorizer(lowercase=False, tokenizer=tokenize, max_features=5000)
        self.tfidf = TfidfTransformer()

    def vec_train(self, data):

        vec = self.vectorizer.fit_transform(data)
        vec_tfidf = self.tfidf.fit_transform(vec)

        return vec, vec_tfidf

    def vec_test(self, data):

        vec = self.vectorizer.transform(data)
        vec_tfidf = self.tfidf.transform(vec)

        return vec, vec_tfidf
      
def create_knn_classifier(vec, labels, k):

    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(vec, labels)

    return clf
