import csv
import sys
import numpy as np
import data_config
import new_classifiers5
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.metrics  import f1_score, mean_squared_error, accuracy_score, confusion_matrix, precision_score, recall_score
import keep_expanding_mean_iteration
import different_expansions_mean_iterations
import baseline2
import time
from sklearn import preprocessing
from collections import Counter
import mean_combine
from sklearn.metrics import normalized_mutual_info_score

def main(): 	
	normalize= True
	choose_all= True
	only_top_labels= True
	training_set3 = True	
	new_features_only = False	
	greedy_find_best = True
        ensemble_methods = False

	if(sys.argv[3]== "True" or sys.argv[3]== "TRUE"):
           skip_line = True
        else:
	   skip_line = False

	if(sys.argv[2]== "True" or sys.argv[2]== "TRUE"):
           binary = True
        else:
	   binary = False

        baseline_needed = False

        if(sys.argv[4]== "True" or sys.argv[4]== "TRUE" ):
            MC3R = True
        else:
            MC3R = False

        if(sys.argv[5]== "True" or sys.argv[5]== "TRUE"):
            MC3S = True
        else:
            MC3S = False
	

	f = None
	opener = sys.argv[1]
	csv_f = None
	
	f = open(opener)
	csv_f = csv.reader(f)
	
	#skip line
	if(skip_line):
		csv_f.next()
		csv_f.next()

	print opener
		
	#Put all the data into array All
	All = np.array(data_config.read_in_full_file(csv_f))
	chosen=0
	if(choose_all==True):
		chosen = len(All[0])-1
	else:
		chosen =12
	
	
	'''
		Call only_first_n to divide data into test, train
		and to combine classifiers.
	'''
	set_holders = data_config.only_first_n(All, chosen, normalize, only_top_labels, training_set3)	
	labels = data_config.determine_labels(All[:,len(All[0])-1])
	lb = None
	if(not(binary)):
		lb = preprocessing.LabelBinarizer()
		lb.fit(labels)
		
	#testing if the split is correct
	#tests.determine_correct_split(set_holders["TR1_outcome"], set_holders["TR2_outcome"], set_holders["TS_outcome"])
	
	
	#Stores whole training set. Can be split into 3 or into 2 depending if training_set3 is true or false
	TR_set = {}
	
	

	
	
	
	'''
	Breakes training set into 3, stored in TR_set
	Uses TR1 to expanded TR3.
	TR3_expanded contains TR3_expanded and the models used to do this.
	'''
	if(training_set3):
		TR_set = new_classifiers5.get_new_training_into3(set_holders["TR_full"])
	
	else:
		TR_set= new_classifiers5.get_new_training(set_holders["TR_full"])
		
	
	TR_set['TS'] = set_holders["TS"]
	TR_set['TS_outcome'] = set_holders["TS_outcome"]
	weight = None
	if(not(binary)):
		weight = Counter(set_holders["TS_outcome"])
		for key in weight: 
			weight[key] = float(weight[key]/float(len(set_holders["TS_outcome"])))

	if(baseline_needed == True):
		baseline2.baseline(TR_set, labels, ensemble_methods, lb, weight, binary)
	
	best_strings_first = None
	if(MC3S):
		classifier_outcomes, best_strings_first = different_expansions_mean_iterations.all(set_holders["TR_full"], TR_set, labels, binary, set_holders["TS"], set_holders["TS_outcome"], training_set3, new_features_only, False, ensemble_methods, weight, lb)
		roc_score = 0
		for outcome in classifier_outcomes:
				if(binary):
		                        if(outcome['type'] == 'weighted majority voting ensemble'):
						roc_score = roc_auc_score(set_holders["TS_outcome"], outcome['prediction'], average= "weighted")
						print roc_score
						print 'MC3-S'
				else:
					if(outcome['type'] == 'weighted majority voting ensemble'):
						roc_score = new_classifiers5.multi_class_roc(weight, lb, outcome['prediction'], set_holders["TS_outcome"], labels)
						print roc_score
						print 'MC3-S'
	
		

		

	if(MC3R):	
                if(not(MC3S)):
                   best_classifiers,best_strings_first = different_expansions_mean_iterations.expand_best(TR_set, labels, binary, new_features_only, training_set3, ensemble_methods, lb, weight)
		classifier_outcomes = keep_expanding_mean_iteration.one_iteration(TR_set,  labels, binary, ensemble_methods, weight, lb, False, False, best_strings_first)
		start_time_cons = time.time()
		classifier_outcomes = mean_combine.combine_census(set_holders["TR_full"], classifier_outcomes['TR_set_used'], set_holders["TS"], set_holders["TS_outcome"], classifier_outcomes['best classifiers'], 
		labels, binary, new_features_only, classifier_outcomes["best strings"],classifier_outcomes["best strings"], ensemble_methods, weight, lb, TR_set, False)
		
		roc_score = 0
		for outcome in classifier_outcomes:
				if(binary):
		                        if(outcome['type'] == 'weighted majority voting ensemble'):
						roc_score = roc_auc_score(set_holders["TS_outcome"], outcome['prediction'], average= "weighted")
						print roc_score
						print 'MC3-R'
				else:
					if(outcome['type'] == 'weighted majority voting ensemble'):
						roc_score = new_classifiers5.multi_class_roc(weight, lb, outcome['prediction'], set_holders["TS_outcome"], labels)
						print roc_score
						print 'MC3-R'
				
	
main()
