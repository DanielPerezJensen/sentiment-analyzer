# sentiment-analyzer

#### A machine learning application of sentiment classifying written in Python3.6.
##### Requirements
1. Python3.6 
2. For pip modules run: *_pip install -r requirements.txt_* or *_pip install nltk scikit-learn scipy_*

By using the 5 classifiers:
1. [Naive Bayes Classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
2. [Multinomial Naive Bayes Classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Multinomial_naive_Bayes)
3. [Bernoulli Naive Bayes Classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Bernoulli_naive_Bayes)
4. [Logistic Regression Classifier](https://en.wikipedia.org/wiki/Logistic_regression)
5. [Linear SVC Classifier](https://en.wikipedia.org/wiki/Support_vector_machine)

All classifiers were trained on 10.000 short reviews of movies that were either positive (pos) or negative (neg). 
By combining the votes of these 5 classifiers a custom Voted Classifier has been constructed. 
This Voted Classifier (VCLF) uses the votes of the 5 trained classifier to come to a classification of a given string.

If you wish to use this classifier for yourself do the following steps:
1. git clone https://github.com/DanielPerezJensen/sentiment-analyzer.git
2. cd sentiment-analyzer
  * If you want to train the classifiers yourself or on other data do this:
    1. To add your own data put positive reviews in "sentiment_data/positive" and negative reviews in "sentiment_data/negative". Separate  every review by a new line.
    2. python3.6 train_clf.py
3. Now you can import sentiment.py in files that are in the *sentiment-analyzer* directory.
4. * Usage:
```python
   >> from sentiment import sentiment
   >> print(sentiment("This movie was great"))
   >> pos 1
   >> print(sentiment("I have had a really bad afternoon to be honest"))
   >> neg 1 
```

##### *_With thanks to Sentdex from [pythonprogramming.net](https://pythonprogramming.net/) for coding examples and the training data._*
