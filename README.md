# Text Classification with arXiv
## About this Project
Our goal is to develop a tool to classify scientific literature (in general) by category accurately and efficiently. Our primary goal is to classify papers within general subject matters (ex. Physics). If sufficient progress is made we would like to aim for our second goal: to classify texts within subject matters (ex. Physics > Astrophysics) and where these texts can belong in multiple subcategories.  

The end user of this project is an academic institution that works with processing large quantities of scientific texts, such as research counucils, scientific journals, or academic libraries. 

Currently there is a lack of ability to accurately categorize or sort scientific text. Being able to do so can ease processes within library systems, peer review systems, and make it easier for those within the academic realm to find relevant scientific texts. 

## About
### The Model

The 'trainAcademicClassifier.py' program trains the model using the arXiv dataset. It takes 1 command line argument, the file name/path to the training data. It produces two files, the model .pkl file and the vectorizer .pkl file, both necesssary for predicting with the model. 
 
Example: 'python trainAcademicClassifier.py training_data.json'
 
Given that these two files have been produced the 'classify.py' program can be run to classify a single text using the model. It also takes one command line argument, the file name/path to the text you want to classify, in .txt format. The program analyzes the text and prints the predicted category.
 
Example: 'python classify.py text.txt'

## Citations
 arXiv.org submitters. (2022). <i>arXiv Dataset</i> [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/4420270
 

