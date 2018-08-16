# sentiment-analyzer

#### A machine learning application of sentiment analyzing written in Python3.6.
##### Requirements
1. Python3.6 
2. For pip modules required run: *_pip install -r requirements.txt_*

By using the 5 classifiers:
1. Naive Bayes Classifier
2. Multinomial Classifier
3. Bernoully Naive Bayes Classifier
4. Logistic Regression Classifier
5. Linear SVC Classifier

All classifiers were trained on 10.000 short reviews of movies that were either positive (pos) or negative (neg). 
By combining the votes of these 5 classifiers a custom Voted Classifier has been construced. 
This Voted Classifier (VCLF) uses the votes of the 5 trained classifier to come to a classification of a given string.

If you wish to use this classifier for yourself do the following steps:
1. git clone https://github.com/DanielPerezJensen/sentiment-analyzer.git
2. cd sentiment-analyzer
3. python3.6 train_clf.py
4. If you get an error create these directories: **pickled/data** and **pickled/algorithms**
5. Now you can import sentiment.py in files that are in the *sentiment-analyzer* directory.
6. * Usage:
   * from sentiment import sentiment
   * print(sentiment("This movie was great"))
   * print(sentiment("I have had a really bad afternoon to be honest"))
   * Output: <sentiment classification> <classification confidence>


##### *_With thanks to Sentdex from [pythonprogramming.net](https://pythonprogramming.net/) for coding examples and the training data._*
