# Spam Classifier
Training a **spam classifier** to differentiate between _spam_ and _ham_ emails. The dataset used for training is the [SpamAssassin public mail corpus](http://spamassassin.apache.org/old/publiccorpus/) which consists of a seleciton of mail messages, labelled as spam or ham.

## Overview
The goal is to train a classification algorithm to differentiate between _spam_ and _ham_ emails. 

To reach the goal, several classification models are first trained on the Apache SpamAssassin dataset. After evaluating each classifier, the 3 best performing one are  fine-tuned and re-evaluated again. The 'best' classifier is then saved as .pkl file.


All the steps are contained and documented in the Jupyter Notebok [Spam classifier](https://github.com/ElisaCovato/Spam-Classifier/blob/main/Spam%20classifier.ipynb)). To lighten the amount of line of codes in the notebook, all the functions needed have been collected in different scripts - the notebook will take care to call and run such functions.



### Classification models
The classifiers used are:
- [SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)
- [MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
- [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)
- [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier)
- [AdaBoostClassfier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier)
- [KNNClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier)
- [NaiveBayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB)
- [(Linear) SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

### Dataset
The above classifier are trained on the [SpamAssassin public mail corpus](http://spamassassin.apache.org/old/publiccorpus/) which consists of a selection of mail messages, labelled as spam or ham.

### Evaluation
Each classifier is evaluated using the following performance measures:
* confusion matrix
* accuracy
* precision
* recall
* f1 score

## How to start
1. Check out that you have a working Python(preferably Python 3) and Jupyter Notebook installation 
2. Git clone the [repo](https://github.com/ElisaCovato/Spam-Classifier.git) and cd inside directory
3. Install requirements: 

    `pip install -r requirements.txt`
4. Check that all the modules have been installed, and download the mail datasets : 
    
    `python startup.py`
5. Use the jupyter notebook [Spam classifier](https://github.com/ElisaCovato/Spam-Classifier/blob/main/Spam%20classifier.ipynb) to download and familiarize yourself with the data, train, evaluate and fine tune the classifiers, and pick the 'best' performing ones.
    
