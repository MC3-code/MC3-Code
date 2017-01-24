from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
import math 
import numpy as np
from sklearn import datasets, linear_model, cross_validation, svm
from sklearn.metrics import accuracy_score, roc_auc_score
import data_config
from sklearn.neural_network import BernoulliRBM
import measures
import copy
import new_classifiers5
import combine5
from operator import itemgetter


def find_best(best_classifiers, TS, TS_outcome, labels, TR, TR_outcome, weight, lb, binary):


	#picks the best classifier based on ROC score on TR3. Appends best and roc score of best
	greedy_best = []
	
	pick = pick_best(best_classifiers)
	best = pick[0]
	
	#left list contains all classifiers except best one.
	left_list = pick[1]
	
	
	greedy_best.append(best)
	greedy_best_score = best[2]
	
	'''
	print greedy_best[0][3]
	print greedy_best[0][3].predict(TS)
	print "worked"
	'''
	
	best_still = True
	while best_still:
		most_diverse_tup = most_diverse_with_list(greedy_best, left_list, TS_outcome, labels)
		most_diverse = most_diverse_tup[0]
		if(not(most_diverse)):
			break
		remove_one = most_diverse_tup[1]
		model_dict = convert_from_tup_to_baseline_dict(most_diverse)
		now_pred = combine5.combine_baseline(model_dict, TS, labels, TR, TR_outcome)
		now_score = 0.0
		if(binary):
			now_score = roc_auc_score(TS_outcome, now_pred, average= 'weighted')
		else:
			now_score = new_classifiers5.multi_class_roc(weight, lb, now_pred, TS_outcome, labels)
		
		
		if(now_score> greedy_best_score):
			greedy_best_score = now_score
			greedy_best.append(remove_one)
			left_list.remove(remove_one)
		else:
			best_still = False
	
	names = []
	for best in greedy_best:
		names.append(best[0])	
	return names
	
	
def pick_best(classifier_store):
	left_list = sorted(classifier_store, key=itemgetter(2)) 
	best = left_list.pop(0)
	return (best, left_list)


'''
Input: 
	greedy_best- List of best classifiers- Each classifier is tuple 
'''	
def most_diverse_with_list(greedy_best, left_list, TS_outcome, labels):
	max_diversity= 0.0
	max_group = None
	best_left_one = None
	
	for left_one in left_list:
		classifier_list = each_combination(greedy_best, left_one)
		cur_diverse = measures.diversity_measure(classifier_list, TS_outcome , labels)
		if(max_diversity < cur_diverse):
			max_diversity = cur_diverse
			max_group = classifier_list
			best_left_one = left_one
			
	return (max_group, best_left_one)

def each_combination(greedy_best, current):
	hold = []
	for each_greedy in greedy_best:
		hold.append(each_greedy)
	hold.append(current)
	return hold
	
	
def convert_from_tup_to_baseline_dict(most_diverse):
	hold = []
	for each_classifier in most_diverse:
		hold.append({'type': each_classifier[0], 'coefficent': each_classifier[2], 'model': each_classifier[3] })
	return hold