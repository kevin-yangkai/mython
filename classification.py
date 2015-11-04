import os, pickle, argparse
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.dummy import DummyClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

_ALL_ = ["random", "decision_trees", "linear_svm", "gaussian_naive_bayes"]

def random(X, y):
	clf = DummyClassifier(strategy="uniform")
	clf.fit(X, y)
	return clf

def decision_trees(X, y, max_depth=25):
	clf = DecisionTreeClassifier(max_depth=max_depth)
	clf.fit(X, y)
	return clf

def linear_svm(X, y):
	clf = OneVsRestClassifier(LinearSVC(random_state=0))
	clf.fit(X, y)
	return clf

def gaussian_naive_bayes(X, y):
	clf = OneVsRestClassifier(GaussianNB())
	clf.fit(X, y)
	return clf

def classify(X, y, algorithm):
	if algorithm == "decision_trees":
		return decision_trees(X,y)
	if algorithm == "linear_svm":
		return linear_svm(X, y)
	if algorithm == "gaussian_naive_bayes":
		return gaussian_naive_bayes(X,y)
	if algorithm == "random":
		return random(X, y)

def classify_and_predict(X, Y, algorithm):
	print "####Classifying and predicting with '%s'...####" % algorithm
	kf = KFold(len(Y), n_folds=4)
	confusion_matrix = np.ndarray(shape=Y.shape)
	for train, test in kf:
		X_train, X_test, y_train, y_test = X[train], X[test], Y[train], Y[test]
		clf = classify(X_train, y_train, algorithm)
		y_predicted = clf.predict(X_test)
		for i in range(len(test)):
			confusion_matrix[test[i]] = y_predicted[i][:]
	return confusion_matrix	

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Classify and predict using cross-validation')
	parser.add_argument('collection', help='Collection name (e.g.: majorminer)')
	parser.add_argument('-a', '--algorithm', nargs='?', default="all", help='algorithm to run (default="all")')
	args = parser.parse_args()
	
	feature_model = pickle.load(open("feature_models/%s.pickle" % args.collection))
	X = feature_model['X']
	Y = feature_model['Y']
	
	if not os.path.exists("classification"):
		os.mkdir("classification")
	if not os.path.exists("classification/%s" % args.collection):
		os.mkdir("classification/%s" % args.collection)

	np.savetxt("classification/%s/ground_truth.out" % args.collection, Y, fmt="%d") # save ground truth
	
	if args.algorithm == "all":
		for algorithm in _ALL_:
			confusion_matrix = classify_and_predict(X, Y, algorithm)
			np.savetxt("classification/%s/confusion_matrix__%s.out" % (args.collection, algorithm), confusion_matrix, fmt="%d")
	else:
		confusion_matrix = classify_and_predict(X, Y, args.algorithm)
		np.savetxt("classification/%s/confusion_matrix__%s.out" % (args.collection, args.algorithm), confusion_matrix, fmt="%d")
	