import dataPrep
import numpy as np
import pandas as pd
import sys
import joblib

def main():
    with open('academicClassifierModel.pkl') as model:
        model = json.load(model)
    with open('academicClassifierVectorizer.pkl') as vectorizer:
        model_vectorizer = vectorizer
    with open(str(sys.argv[0])) as input:
        input_text = input
    categorydict = loadCategoryDict('categoryKeys.json')
    input_text.str.lower()
    input_text.str.replace(r'[^\w\s]+', '', regex=True)
    stop = stopwords.words('english')
    ' '.join([word for word in input_text.split() if word not in (stop)])
    ' '.join([wnl.lemmatize(word) for word in input_text.split()])
    input_text = np.array([input_text])
    input_vector = model_vectorizer.transform(input_text)
    prediction = model.predict(input_vector)[0]
    print(prediction)

if __name__ == "__main__":
    main()