import math 
import numpy as np
import data_config
import copy
import new_classifiers5
import itertools
import mean_combine
import greedy
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.grid_search import GridSearchCV
'''
Input:
	full_training_set- Full training set
	TR_set_used- Contains dict of full training set split into 3 parts- 
				TR1 + TR1_outcome, TR2  + TR2_outcome, TR3 + TR3_outcome 
				(6 different dicts)
	labels- List of all the class labels in the data
	binary- Boolean, true if the class labels have only 2 options, false otherwise
	TS- The testing set data without the outcomes. (the real test data)
	TS_outcome- The testing set data with the outcomes. (the real test outcomes)
	training_set3- If training set split into 3 or 2.
	new_features_only- Boolean. True if only the predictions of the classifiers included in features, false otherwise
	greedy_find_best- Boolean. True if using a greedy approach to determine best subset of classifiers, false otherwise. 
	
Description:
	
Output:
	Returns tuple of 4 different expansions of algorthim:
		1) Expand best Combine all 
		2) Expand best Combine best
		3) Expand all Combine best
		4) Expand all Combine all
	
	Each tuple contains a list of dicts.
	Dicts contains:
		type- type of classifier 
		prediction- contains prediction on TS
		(also includes ensemble as one type)
	
'''
def all(full_training_set, TR_set_used, labels, binary, TS, TS_outcome, training_set3, new_features_only, greedy_find_best, ensemble_methods, weight, lb):
	#Training set used to determine best classifier*. 
	keeper = expand_best(TR_set_used, labels, binary, new_features_only, training_set3, ensemble_methods, lb, weight)
	best_classifiers = keeper[0]
	best_strings_first = keeper[1]
	
	hold3 = expand_all_combine_all(full_training_set, TR_set_used, TS, TS_outcome, best_classifiers, labels, binary, new_features_only, best_strings_first, ensemble_methods, weight, lb)
	
	return (hold3, best_strings_first)
	
'''
Input- 
	TR_set_used- Contains dict of TR1, TR1_outcome, TR2, TR2_outcome, TR3, and TR3_outcome.

Description: 

	Different from expand_best. 
	Does not find subset of classifiers used to create expanded set.
	Instead uses all classifiers to create expanded set.

	1)Stores in best_strings_first- all the classifiers in a list of strings (SVM, Random Forest,....etc)
	2) Creates best classifiers- Array of dicts where each classifiers prediction on the expanded set (created by the all of classifers' prediction). Each dict contains: mean_score, model2, type, coefficent, and prediction

Return-
	best classifers- talked about above
	best_strings_first- all the classifers in list of strings representation. 
'''	

def expand_best(TR_set_used, labels, binary, new_features_only, training_set3, ensemble_methods, lb, weight):
    
    #for each classifier train on tr and test on tR3
    #
    list_classifiers = [] 
    for each_classifier in new_classifiers5.create_classifiers(ensemble_methods):

        model = None
        if(each_classifier['tuned_parameters'] != []):
            model = GridSearchCV(each_classifier['model'], each_classifier['tuned_parameters'], cv=10, scoring="accuracy").fit(TR_set_used['TR'],TR_set_used['TR_outcome'])

        else:
            model = each_classifier['model'].fit(TR_set_used['TR'],TR_set_used['TR_outcome'])


        type_hold = each_classifier['type']
        predictions = model.predict(TR_set_used['TR3'])
        
	roc_score = None
        if(binary):
            roc_score = roc_auc_score(predictions,TR_set_used['TR3_outcome'] )
        else:
            roc_score = new_classifiers5.multi_class_roc(weight, lb, predictions, TR_set_used['TR3_outcome'], labels)           

        hold_tup = (type_hold, predictions, roc_score, model)
        list_classifiers.append(hold_tup)

    best_strings_first = greedy.find_best(list_classifiers, TR_set_used['TR3'], TR_set_used['TR3_outcome'], labels, TR_set_used['TR'],TR_set_used['TR_outcome'], weight, lb, binary)

    
    best_strings_second = new_classifiers5.names_all_classifiers(ensemble_methods)

    best_classifiers = new_classifiers5.one_iteration(TR_set_used, training_set3, new_features_only, labels, binary, best_strings_first, best_strings_second, ensemble_methods, lb, weight)
   
    return (best_classifiers, best_strings_first)

    #(type, prediction, roc_auc_score, model)

def expand_all(TR_set_used, labels, binary, new_features_only, training_set3, ensemble_methods, lb, weight):

        best_strings_first = new_classifiers5.names_all_classifiers(ensemble_methods)
	best_strings_second = best_strings_first

	best_classifiers = new_classifiers5.one_iteration(TR_set_used, training_set3, new_features_only, labels, binary, best_strings_first, best_strings_second, ensemble_methods, lb, weight)
	
	return (best_classifiers, best_strings_first)


def expand_all_combine_all(full_training_set, TR_set_used, TS, TS_outcome, best_classifiers, labels, binary, new_features_only, best_strings_first, ensemble_methods, weight, lb):
	best_strings_second = new_classifiers5.names_all_classifiers(ensemble_methods)
	return mean_combine.combine(full_training_set, TR_set_used, TS, TS_outcome, best_classifiers, labels, binary, new_features_only, best_strings_first, best_strings_second, ensemble_methods, weight, lb)


