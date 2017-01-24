from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
import math 
import numpy as np
from sklearn import datasets, linear_model, cross_validation, svm
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import data_config
import csv
import numpy as np
from collections import Counter
import math 
from sklearn.cross_validation import StratifiedShuffleSplit
from operator import itemgetter, attrgetter    
import new_classifiers5
import itertools
import sys

#array of tuples consisting of type, predictions, weight
def diversity_measure(classifier_list, TR_outcome, labels):
	#classifier list tuple of weight and classifier for each classifier.
	#test case 1	
	#total weight of all the classes
	total_weight =0
	for y in range(0,len(classifier_list)):
		total_weight = total_weight + classifier_list[y][2]
		
	total = 0
	
	
	#loop through each 
	for m in range (0, len(TR_outcome)):
		all_labels = 0
		#loop through each label
		for y in range(0, len(labels)):
			sum = 0			
			for j in range(0,len(classifier_list)):
				if(classifier_list[j][1][m]==labels[y]):
					sum = sum + classifier_list[j][2]
					
			all_labels = all_labels + math.pow(float(sum/total_weight), 2)
		total = (total) + (1-all_labels)

	return float(total/len(TR_outcome))

'''
Creates an array of tuples. Where each tuple is given by (type, prediction, roc_auc_score, model)
'''
def create_all(TR1, TR1_outcome, TR2, TR2_outcome, labels, binary, ensemble_methods, weight, lb):
	classifier_list = []
	for each_classifier in new_classifiers5.create_classifiers(ensemble_methods):
		store = new_classifiers5.tuned_classifier(TR1, TR1_outcome, TR2, each_classifier['model'], 
			each_classifier['tuned_parameters'], each_classifier['type'], ensemble_methods)
		tuple_hold = None
		store['prediction'] = np.array(store['prediction'])
		prediction = []
		for each_one in store['prediction']:
			prediction.append(each_one.argmax(axis=0))
			
		if(binary):
			tuple_hold = (store['type'], prediction, roc_auc_score(TR2_outcome, prediction, average= 'weighted'), store['model'])
		else:
			tuple_hold = (store['type'], prediction, new_classifiers5.multi_class_roc(weight, lb, prediction, TR2_outcome, labels), store['model'])
		classifier_list.append(tuple_hold)

	return classifier_list
	
	
	
def find_max_min(classifiers, TR2_outcome, labels):
	list2 = [0, 1, 2, 3, 4, 5, 6, 7 ]
	max_average = float(0)
	min_average = float(sys.maxint)
	
	max_diversity = float(0)
	min_diversity = float(sys.maxint)
	
	max_diversity_sub = None
	max_average_sub = None
	
	for L in range(1, 9):
		for subset in itertools.combinations(list2, L):
			classifier_list= []
			#Take a subset of the classifiers given by subset 
			append_all_numbers(classifier_list, classifiers, subset)
			
			'''
			uses the value measure to determine best subset:
					1)Combination of average ROC score of subset of classifiers
					2)Diversity measure of the subset of classifiers
			'''
			average_value = average(classifier_list)
			diversity_value = diversity_measure(classifier_list, TR2_outcome, labels)
			'''
			print convert_to_models(subset, classifiers)
			print average_value
			print " "
			print diversity_value
			print " "
			print " "
			'''
			if(average_value> max_average):
				max_average = average_value
				max_average_sub = subset
			elif(average_value< min_average):
				min_average = average_value
				
			if(diversity_value> max_diversity):
				max_diversity = diversity_value
				max_diversity_sub = subset
			elif(diversity_value< min_diversity):
				min_diversity = diversity_value
	print max_diversity_sub
	return {'max diversity': max_diversity, 'min diversity': min_diversity, 'max average': max_average, 'min average': min_average, "diverse_subset":convert_to_models(max_diversity_sub, classifiers)}
	
	
	
def normalize_val(val, max, min):
	return float((val-min)/(max-min))
	
'''
Input:
	classifiers: list of all the classifiers- each classifier is tuple of (type, prediction, roc_auc_score)
	TR2_outcome: the set of data we are testing diversity on
	labels- List of all the class labels in the data
	binary- Boolean, true if the class labels have only 2 options, false otherwise
	
Return:
	Sring represenation of the "best" classifiers (i.e. (Random Forest, SVM, Decision Tree) )

'''
def all_combinations(classifiers, TR2_outcome, labels, binary):
	
	
	max_diversity = 0.0
	max_type = None
	max_average = 0.0
	
	#Each number represents one of the 8 classifiers
	list2 = [0, 1, 2, 3, 4, 5, 6, 7 ]
	
	#find max and min values for diversity and average in order to normalize
	hold_max_min = find_max_min(classifiers, TR2_outcome, labels)
	best_val = 0
	
	#Take combinaitions 1 at a time, 2 at a time, ....6 at a time
	for L in range(1, 9):
		for subset in itertools.combinations(list2, L):
			classifier_list= []
			#Take a subset of the classifiers given by subset 
			append_all_numbers(classifier_list, classifiers, subset)
			
			'''
			uses the value measure to determine best subset:
					1)Combination of average ROC score of subset of classifiers
					2)Diversity measure of the subset of classifiers
			'''
			normalized_average = normalize_val(average(classifier_list), hold_max_min['max average'], hold_max_min['min average'])
				
				
			normalized_diversity = normalize_val(diversity_measure(classifier_list, TR2_outcome, labels), hold_max_min['max diversity'], hold_max_min['min diversity'])

			value = normalized_average + normalized_diversity
			
			if(value> best_val):
				best_val = value
				max_average = normalized_average
				max_diversity = normalized_diversity
				max_type = subset
	return convert_to_models(max_type, classifiers)

'''
Input: 
	classifier_list- List of subset of classifiers
Return:
	The average ROC_AUC_Score of the subset of classifiers
'''
def average(classifier_list):
	sum = 0.0
	for each_classifier in classifier_list:
		sum = sum + each_classifier[2] 
	return sum/len(classifier_list)

'''
Input: 
	keeper- the list which will contain the subset of classifiers.
	classifiers- List of all the classifiers
	subset- contains the indices of the classifiers which will be contained in the subset
	
Puts the subset of classifiers in keep.	
'''
def append_all_numbers(keeper, classifiers, subset):
	for i in range(len(subset)):
		keeper.append(classifiers[subset[i]])
	
'''
Input:
	max_type- contains the indices of the classifiers which will be contained in the "best" subset
	classifiers- List of all the classifiers
	
Returns:
	best_classifiers- String represenation of the "best" classifiers (i.e. (Random Forest, SVM, Decision Tree) )
'''
def convert_to_models(max_type, classifiers):
	best_classifiers = ()
	for i in range(len(max_type)):
		best_classifiers = best_classifiers + (classifiers[max_type[i]][0],)
	return best_classifiers 
	