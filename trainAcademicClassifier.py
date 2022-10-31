import dataPrep
import sys
from sklearn.linear_model import LogisticRegression
import joblib

def main():
    try:
        cleaned_data = cleanData(str(sys.argv[0]))
    except ValueError:
        print('Error loading training data file. Pass training data as .json file in command line') 
    else:
        X_train, __Xtest__, y_train, __ytest__, vectorizer = createTrainingData(cleaned_data)
        model = LogisticRegression(penalty='l2', tol=1e-4, C=1.0, solver='lbfgs', max_iter=500, multi_class='ovr')
        model.fit(X_train, y_train)
        joblib.dump(model, 'academicClassifierModel.pkl')
        joblib.dump(vectorizer, 'academicClassifierVectorizer.pkl')
        
if __name__ == "__main__":
    main()
