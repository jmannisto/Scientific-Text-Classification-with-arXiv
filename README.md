# Text Classification with arXiv
## About this Project
Our goal is to develop a tool to classify scientific literature (in general) by category accurately and efficiently. Our primary goal is to classify papers within general subject matters (ex. Physics). If sufficient progress is made we would like to aim for our second goal: to classify texts within subject matters (ex. Physics > Astrophysics) and where these texts can belong in multiple subcategories.  

The end user of this project is an education institution. 

Currently there is a lack of ability to accurately categorize or sort scientific text. Being able to do so can ease processes within library systems, peer review systems, and make it easier for those within the academic realm to find relevant scientific texts. 

## About
<details><summary>The Model</summary>
<p>
The 'trainAcademicClassifier.py' program trains the model using the arXiv dataset. It takes 1 obligatory and one optional command line argument. First, the file name/path to the training data, then optionally the number of samples (the default value being 20,000). It produces two files, the model .pkl file and the vectorizer .pkl file, both necesssary for predicting with the model. 
 
Example: 'python trainAcademicClassifier.py training_data.json 30000'
 
Given that these two files have been produced the 'classify.py' program can be run to classify a single text using the model. It takes one command line argument, the file name/path to the text you want to classify, in .txt format. The program analyzes the text and prints the predicted category.
 
Example: 'python classify.py text.txt'
</p>
</details>

<details><summary>The Data</summary>
<p>

</p>
</details>

## Citations
 arXiv.org submitters. (2022). <i>arXiv Dataset</i> [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/4420270
 

