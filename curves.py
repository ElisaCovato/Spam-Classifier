from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt

import numpy as np



class PlotCurves():

	def __init__(self, y_preds, X_train, y_train):
		self.y_preds = y_preds
		self.X_train = X_train
		self.y_train = y_train
		self.plot_style = styles = {
			'SGD': 'r--',
			'MLP': 'm--',
			'Decision Tree': 'y--',
			'Random Forest': 'g--',
			'AdaBoost': 'b--',
			'KNN': 'k--',
			'NB': 'p--',
			'SVM': 'c--'
		}

		# define classifier
		sgd_clf = SGDClassifier(random_state=42, max_iter=100)
		mlp_clf = MLPClassifier(hidden_layer_sizes=(16,))
		tree_clf = DecisionTreeClassifier()
		forest_clf = RandomForestClassifier()
		adaboost_clf = AdaBoostClassifier()
		knn_clf = KNeighborsClassifier()
		nb_clf = GaussianNB()
		svm_clf = SVC()
		self.clf = {
			'SGD': sgd_clf,
			'MLP': mlp_clf,
			'Decision Tree': tree_clf,
			'Random Forest': forest_clf,
			'AdaBoost': adaboost_clf,
			'KNN': knn_clf,
			'NB': nb_clf,
			'SVM': svm_clf
		}


	def precision_vs_recall(self, clf_list, thr):
		if clf_list == 'all':
			clf_list = self.clf.keys()


		# plot precision versus recall curve for each classifier
		for clf in clf_list:
			# compute decision scores using cross valuation
			y_scores = self.compute_scores(clf)
			precisions, recalls, thresholds = precision_recall_curve(self.y_train, y_scores)

			# plot the precision and recall curves
			plt.plot(recalls, precisions, self.plot_style[clf], label=clf)

			if thr == 'default':
				thr = 0
			if thr == 'best':
				thr= thresholds[np.argmax((recalls <= 0.9))]

			# higlight threshold 
			y_pred_thr = y_scores >= thr
			hl_precision = precision_score(self.y_train, y_pred_thr)
			hl_recall = recall_score(self.y_train, y_pred_thr)
			plt.plot([0, hl_recall], [hl_precision, hl_precision], 'r:')
			plt.plot([hl_recall, hl_recall], [0, hl_precision], 'r:')
			plt.plot([hl_recall], [hl_precision], 'ro')

		# style plot
		plt.grid(True)
		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.legend()
		plt.show()

	def roc_curve(self, clf_list, thr):
		if clf_list == 'all':
			clf_list = self.clf.keys()

		for clf in clf_list:
			# compute decision scores using cross valuation
			y_scores = self.compute_scores(clf)
			# fpr = false positive rate = recall/sensitivity
			# tpr = true positive rate
			fpr, tpr, thresholds = roc_curve(self.y_train, y_scores)

			# plot the roc curve
			plt.plot(fpr, tpr, self.plot_style[clf], label=clf)

			if thr == 'default':
				thr = 0
			if thr == 'best':
				thr = thresholds[np.argmax((recalls <= 0.9))]

			y_pred = y_scores >= thr
			fp = np.sum(np.logical_and(y_pred == True, self.y_train == 0))
			tp = np.sum(np.logical_and(y_pred == True, self.y_train == 1))
			fpr = fp / np.sum(self.y_train == 0)
			tpr = tp / np.sum(self.y_train == 1)    
			plt.plot([0, fpr], [tpr, tpr], 'r:')
			plt.plot([fpr, fpr], [0, tpr], 'r:')
			plt.plot([fpr], [tpr], 'ro')

		# style plot
		plt.grid(True)
		plt.xlabel('Specificity')
		plt.ylabel('Recall/sensitivity')
		plt.legend()
		plt.show()

	def roc_auc_score(self, clf_list):
		if clf_list == 'all':
			clf_list = self.clf.keys()

		print("ROC AUC scores:")
		for clf in clf_list:
			y_scores = self.compute_scores(clf)
			roc_auc = roc_auc_score(self.y_train, y_scores)
			print("{}: {}".format(clf, roc_auc))






	def compute_scores(self, classifier):
		classifier = self.clf[classifier]
		method = 'decision_function'
		if not hasattr(classifier, 'decision_function') and hasattr(classifier, 'predict_proba'):
			method = 'predict_proba'
		y_scores = cross_val_predict(classifier, self.X_train, self.y_train, 
			cv=3, method=method)

		if method == 'predict_proba':
			y_scores = y_scores[:,1]

		return y_scores



