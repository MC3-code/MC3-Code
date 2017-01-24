import csv
import numpy as np
from collections import Counter
import math 
from sklearn.cross_validation import StratifiedShuffleSplit
'''
	Returns all the class labels used by current array. 
'''
def determine_labels(column_matrix):
	labels= []
	for x in range(0, len(column_matrix)):
		if not(column_matrix[x] in labels):
			labels.append(column_matrix[x])
	return sorted(labels)
	
'''
	Divides array All into two arrays with equal distributin of class
	labels. 
'''	
def create_balanced_sets(All, size):
	y = (All[:,len(All[0])-1])
	hold =StratifiedShuffleSplit(y , 10, test_size = size)	
	t1 = []
	t2 = []
	train = None
	
	for train_index, test_index in hold:
		t1, t2 = All[train_index], All[test_index]
	t1 = np.array(t1)
	t2 = np.array(t2)
	
	if(size ==0.5):
		train = np.concatenate((t1, t2), axis=0)
	return (t1,t2, train)
	

'''
	Given an array of features and class labels, keeps 
	all data that has class labels >= 11.
	Returns a new array removing all data with class label <11.
'''
def top_classes(All):
	labels = determine_labels((All[:,len(All[0])-1]))
	tops = (All[:,len(All[0])-1]).tolist()
	keep = {k: v for k, v in Counter(tops).iteritems() if v>=11}
	keys = keep.keys()
	keep_indices = []
	y = (All[:,len(All[0])-1])
	for i in range(0, len(y)):
		if y[i] in keys:
			keep_indices.append(i)
	All = All[keep_indices]
	return All

'''
	Normalize each column of array by dividing
	the max in each column by each element in column. 
'''
def normalize(All, number_column):
	abe = np.array(All)
	store = abe[:,0].astype(float)
	store = store/np.amax( abe[:,0])
	for x in range(1, number_column):
		hold = abe[:,x].astype(float)
		if(np.amax( abe[:,x])!=0):
			hold = hold/np.amax( abe[:,x])
		store = np.column_stack((store, hold))
	for x in range(number_column, (len(All[0]))):
		hold = abe[:,x].astype(float)
		store = np.column_stack((store, hold))

	return store

'''
Takes in an array with all contents of file.
Normalizes all columns (uses normalize method)
Removes cases where class label is less than a certain threshold (uses top_classes method)
Breaks array with all contents into balanced sets (uses create_balanced_sets method):
	testing
	2 training sets
	outcomes.
Returns these arrays in a list
'''
def only_first_n(All, first_n, normalization, only_top_labels, training_set3):
	if(normalization):
		All = normalize(All, first_n)
	if(only_top_labels):
		All = top_classes(All)

	#Break into 2 training sets and testing set.
	train, test, none = create_balanced_sets(All, 0.2)
	TR1, TR2, TR =create_balanced_sets(train, 0.5)
	
	
	
	
	#Break testing, and two training sets into outcome, and features.
	TR1_outcome = TR1[:,len(TR1[0])-1]
	TR1 = TR1[:,:first_n]
	
	TR2_outcome = TR2[:,len(TR2[0])-1]
	TR2 = TR2[:,:first_n]
	
	TR_outcome = train[:,len(train[0])-1]
	TR = train[:,:first_n]
	
	TS_outcome = test[:,len(test[0])-1]
	TS = test[:,:first_n]
	
	#contains all sets. 
	all_sets = {"TR1" : TR1, "TR1_outcome" : TR1_outcome, "TR2" :TR2, "TR2_outcome": TR2_outcome, 
		"TR": TR, "TR_outcome": TR_outcome, "TS" : TS, "TS_outcome": TS_outcome, "TR_full": train}
	
	return all_sets
	
	
def only_first_n_pick(All, first_n, normalization, only_top_labels, training_set3, test_set):
	if(normalization):
		All = normalize(All, first_n)
	if(only_top_labels):
		All = top_classes(All)

	#Break into 2 training sets and testing set.
	train, test, none = create_balanced_sets(All, test_set)
	TR1, TR2, TR =create_balanced_sets(train, 0.5)
	
	
	
	
	#Break testing, and two training sets into outcome, and features.
	TR1_outcome = TR1[:,len(TR1[0])-1]
	TR1 = TR1[:,:first_n]
	
	TR2_outcome = TR2[:,len(TR2[0])-1]
	TR2 = TR2[:,:first_n]
	
	TR_outcome = train[:,len(train[0])-1]
	TR = train[:,:first_n]
	
	TS_outcome = test[:,len(test[0])-1]
	TS = test[:,:first_n]
	
	#contains all sets. 
	all_sets = {"TR1" : TR1, "TR1_outcome" : TR1_outcome, "TR2" :TR2, "TR2_outcome": TR2_outcome, 
		"TR": TR, "TR_outcome": TR_outcome, "TS" : TS, "TS_outcome": TS_outcome, "TR_full": train}
	
	return all_sets
	

def only_first_n_four_tests(All, first_n, normalization, only_top_labels, training_set3):
	if(normalization):
		All = normalize(All, first_n)
	if(only_top_labels):
		All = top_classes(All)

	#Break into 2 training sets and testing set.
	test, train = create_balanced_sets(All, 0.6)
	ts1, test,  =create_balanced_sets(test, 0.75)
	test, ts2 = create_balanced_sets(test, 0.333)
	test, ts3 = create_balanced_sets(test, 0.5)
	ts4 = test
	
	
	#Break testing, and two training sets into outcome, and features.
	
	TR_outcome = train[:,len(train[0])-1]
	TR = train[:,:first_n]
	
	ts1_outcome = ts1[:,len(ts1[0])-1]
	ts1 = ts1[:,:first_n]
	
	ts2_outcome = ts2[:,len(train[0])-1]
	ts2 = ts2[:,:first_n]
	
	ts3_outcome = test[:,len(ts3[0])-1]
	ts3 = test[:,:first_n]
	
	ts4_outcome = test[:,len(ts4[0])-1]
	ts4 = test[:,:first_n]
	
	#contains all sets. 
	all_sets = {"TR" : TR, "TR_outcome" : TR_outcome, "ts1" : ts1, "ts1_outcome": ts1_outcome, 
		"ts2" : ts2, "ts2_outcome": ts2_outcome, "ts3": ts3, "ts3_outcome": ts3_outcome, 
		"ts4" : ts4, "ts4_outcome": ts4_outcome, "TR_full": train}
	
	return all_sets
	
#Given a csv reader of file f, returns contents in array All of file f.
def read_in_full_file(csv_f):
	All =[]
	counter=0
	for row in csv_f:
		all_d = []
		add=True
		for i in range(len(row)):
			if(row[i]==''):
				add = False
				break
			else:
				all_d.append(float(row[i]))
		if(add):
			counter = counter +1
			All.append(all_d)
	return All