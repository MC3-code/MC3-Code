from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
import math 
import numpy as np
from sklearn import datasets, linear_model, cross_validation, svm
from sklearn.metrics import accuracy_score
import data_config
import new_classifiers5
import keep_expanding_mean_iteration
from sklearn.metrics  import f1_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import BaggingClassifier
import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import normalized_mutual_info_score
import time


def create_TS_expanded(TR, TR_outcome, TS, TS_outcome, new_features_only, best_strings_first, ensemble_methods):
	expanded_sets = new_classifiers5.create_expanded_sets(TR, TR_outcome, TS, TS_outcome, new_features_only, best_strings_first,ensemble_methods )
	return expanded_sets['TR2_pred']
	
	
	
	
	
def predict_prob_TR3(best_classifiers, TR, TR_outcome, TS, TS_outcome, new_features_only, binary, best_strings_first, best_strings_second, labels, ensemble_methods):
	predict_holder = []
	TS_expanded = create_TS_expanded(TR, TR_outcome, TS, TS_outcome, new_features_only, best_strings_first, ensemble_methods)
	for value in best_classifiers:
		if(value['type'] in best_strings_second):
			coor = None
			if(not(binary)):
				coor = normalized_mutual_info_score(value['model2'].predict(TS_expanded), TS_outcome)
			else:
				TR_coor = np.column_stack((value['model2'].predict(TS_expanded), TS_outcome))
				coor = np.corrcoef(TR_coor.T)
				coor = coor[0][1]
			value["coefficent"] = coor
	
	
	

'''
Description: 
	1) Creates expanded testing set using TR, TR_outcome with classifiers given by best_strings_first.
	2) Returns a list of dicts-
		for each classifier in best_string_seconds create dict of type, prob. of TS_expanded,  prediction of TS_expanded,
			and coefficient.

'''
	
def predict_prob_classifiers(best_classifiers, TR, TR_outcome, TS, TS_outcome, new_features_only, binary, best_strings_first, best_strings_second, labels, ensemble_methods):
	predict_holder = []
	TS_expanded = create_TS_expanded(TR, TR_outcome, TS, TS_outcome, new_features_only, best_strings_first, ensemble_methods)
	for value in best_classifiers:
		if(value['type'] in best_strings_second):
			if(value['type']=="SVM" or value['type']=="Decision Tree" or value['type']=="SDG"):
				clf_isotonic = CalibratedClassifierCV(value['model2'], cv='prefit', method='sigmoid').fit(value['fit_x'], value['fit_y'])
				prediction = np.array(clf_isotonic.predict_proba(TS_expanded))
				predict_holder.append({'type': value['type'], "prob": prediction, 
					"prediction": value['model2'].predict(TS_expanded)})	
					
			else:
				bagging = None
				predict_holder.append({'type': value['type'], "prob": value['model2'].predict_proba(TS_expanded), "prediction": value['model2'].predict(TS_expanded), "model": value['model2']})
	return predict_holder	
		
		
'''
Input: 
	full_training_set- Full training set
	TS- The testing set data without the outcomes.
	TS_outcome- The testing set data with the outcomes.
	best_classifiers- Array of dicts of all classifiers (or only the one in best_strings_second).
					  Each dict contains: type, model2, and coefficent key values. 
	labels- List of all the class labels in the data
	binary- Boolean, true if the class labels have only 2 options, false otherwise
	new_features_only- Boolean. True if only the predictions of the classifiers included in features, false otherwise
	best_strings_first- An list of strings representing the "best" strings used to do the first iteration of prediction and appending to data.
	best_strings_second- A list of strings representing the "best" srings used to do the second iteration, or the ensemble.
	
	
Output: 
	Returns a list of dicts.
		Dicts contains:
			type- type of classifier 
			prediction- contains prediction on TS
		(also includes ensemble as one type)
'''
		
def combine(full_training_set, TR_set_used, TS, TS_outcome, best_classifiers, labels, binary, new_features_only, best_strings_first, best_strings_second, ensemble_methods, weight, lb):
	
	start_time = time.time()
	
	TR = TR_set_used["TR_complete"]
	
	TR_outcome = TR_set_used["TR_complete_outcome"]
	
	predict_prob_TR3(best_classifiers, TR_set_used['TR'], TR_set_used['TR_outcome'], TR_set_used['TR3'], TR_set_used['TR3_outcome'], new_features_only, binary, best_strings_first, best_strings_second, labels, ensemble_methods)
	
	
	classifier_outcomes = []
	
	normalization_factor = 0.0
	
	for classifier_prediction in best_classifiers:
		normalization_factor = normalization_factor + classifier_prediction['coefficent']
	
	
	full_TR3 = np.column_stack((TR_set_used['TR3'], TR_set_used['TR3_outcome']))
	full_TR3 =new_classifiers5.get_new_training(full_TR3)
	
	
	TR_new_setter =copy.deepcopy(TR_set_used)
	
	TR_new_setter['TR1'] =TR_new_setter['TR1'].tolist()
	TR_new_setter['TR2'] = TR_new_setter['TR2'].tolist()
	
	for row in full_TR3['TR1']:
		TR_new_setter['TR1'].append(row)
	
	for row1 in full_TR3['TR2']:
		TR_new_setter['TR2'].append(row1)
	
	TR_new_setter['TR1'] = np.array(TR_new_setter['TR1'])
	TR_new_setter['TR2'] = np.array(TR_new_setter['TR2'])
	TR_new_setter['TR1_outcome'] =np.append(TR_new_setter['TR1_outcome'], full_TR3['TR1_outcome'])
	TR_new_setter['TR2_outcome'] =np.append(TR_new_setter['TR2_outcome'], full_TR3['TR2_outcome'])
	
	
	start_time = time.time()
	best_classifiers1 = new_classifiers5.one_iteration(TR_new_setter, False, new_features_only, labels, binary, best_strings_first, best_strings_second, ensemble_methods, weight, lb)
	
	
	start_time = time.time()
	
	prediction_holder1 = predict_prob_classifiers(best_classifiers1, TR, TR_outcome, TS, TS_outcome, new_features_only, binary, best_strings_first, best_strings_second, labels, ensemble_methods)
	
	for classifier_pred in prediction_holder1:
		classifier_outcomes.append({'type': classifier_pred['type'], 'prediction': classifier_pred['prediction']})
		
	sorted_labprob = np.zeros((len(TS),len(labels)))
	
	#stores the answer for each test sample.	
	answer = np.zeros(len(TS))
	
	counter = 0
	for classifier_prediction in prediction_holder1:
		type = classifier_prediction['type']
		norm_prob = float(best_classifiers[counter]['coefficent']/float(normalization_factor))
		'''
		adds the probabilities of what is in sorted_labprob with predictions
		for ith classifier. 
		'''
		each_ts( classifier_prediction["prob"], sorted_labprob, norm_prob)
		counter = counter +1
		
	for x in range(0, len(sorted_labprob)):
		index = np.argmax(sorted_labprob[x])
		answer[x]= answer[x]+ labels[index]
	
	
	'''
		Finds the class with the largest probability for each test sample
		and stores it in answer.
	'''

	classifier_outcomes.append({"type": "weighted majority voting ensemble", 'prediction': answer, "coefficient": None})
	return classifier_outcomes

	
	
	
'''
Description: 
	1) Creates expanded testing set using TR, TR_outcome with classifiers given by best_strings_first.
	2) Returns a list of dicts-
		for each classifier in best_string_seconds create dict of type, prob. of TS_expanded,  prediction of TS_expanded,
			and coefficient.

'''
	
def predict_prob_classifiers_census(best_classifiers, binary, best_strings_second, TS_expanded):
	predict_holder = []
	measure_diversity = []
	measure_diversity_regular = []
	for value in best_classifiers:
		if(value['type'] in best_strings_second):
			if(value['type']=="SVM" or value['type']=="Decision Tree" or value['type']=="SDG"):
				clf_isotonic = CalibratedClassifierCV(value['model2'], cv='prefit', method='sigmoid').fit(value['fit_x'], value['fit_y'])
				prediction = np.array(clf_isotonic.predict_proba(TS_expanded))
				predict_holder.append({'type': value['type'], "prob": prediction, 
					"prediction": value['model2'].predict(TS_expanded), "coefficent": value["coefficent"]})	
					
			else:
				predict_holder.append({'type': value['type'], "prob": value['model2'].predict_proba(TS_expanded), "prediction": value['model2'].predict(TS_expanded), "coefficent": value["coefficent"], "model": value['model2']})
	
	return predict_holder		
	
	
'''
Input: 
	full_training_set- Full training set
	TS- The testing set data without the outcomes.
	TS_outcome- The testing set data with the outcomes.
	best_classifiers- Array of dicts of all classifiers (or only the one in best_strings_second).
					  Each dict contains: type, model2, and coefficent key values. 
	labels- List of all the class labels in the data
	binary- Boolean, true if the class labels have only 2 options, false otherwise
	new_features_only- Boolean. True if only the predictions of the classifiers included in features, false otherwise
	best_strings_first- An list of strings representing the "best" strings used to do the first iteration of prediction and appending to data.
	best_strings_second- A list of strings representing the "best" srings used to do the second iteration, or the ensemble.
	
	
Output: 
	Returns a list of dicts.
		Dicts contains:
			type- type of classifier 
			prediction- contains prediction on TS
		(also includes ensemble as one type)
'''
		
def combine_census(full_training_set, TR_set_used, TS, TS_outcome, best_classifiers, labels, binary, new_features_only, best_strings_first, best_strings_second, ensemble_methods, weight, lb, TR_beg_set, hamming):
	start_time_cons = time.time()
	TR = full_training_set[:,:len(full_training_set[0])-1]
	
	TR_outcome = full_training_set[:,len(full_training_set[0])-1]
	
	prediction_holder_TR3 = predict_prob_classifiers_census(best_classifiers, binary, best_strings_second, TR_set_used['TR3'])
	
	
	classifier_outcomes = []
	
	normalization_factor = 0.0
	
	for classifier_prediction in best_classifiers:
		normalization_factor = normalization_factor + classifier_prediction['coefficent']
	
	
	full_TR3 = np.column_stack((TR_beg_set['TR3'], TR_beg_set['TR3_outcome']))
	full_TR3 =new_classifiers5.get_new_training(full_TR3)
	
	TR_new_setter =copy.deepcopy(TR_beg_set)
	TR_new_setter['TR1'] =TR_new_setter['TR1'].tolist()
	TR_new_setter['TR2'] = TR_new_setter['TR2'].tolist()
	TR_new_setter['TR'] = TR_new_setter['TR'].tolist()
	
	
	
	for row in full_TR3['TR1']:
		TR_new_setter['TR1'].append(row)
	
	for row1 in full_TR3['TR2']:
		TR_new_setter['TR2'].append(row1)
		
	
	TR_new_setter['TR'] = copy.deepcopy(TR_new_setter['TR1'])
	for row in TR_new_setter['TR2']:
		TR_new_setter['TR'].append(row)
	
	
	TR_new_setter['TR1'] = np.array(TR_new_setter['TR1'])
	TR_new_setter['TR2'] = np.array(TR_new_setter['TR2'])
	TR_new_setter['TR'] = np.array(TR_new_setter['TR'])
	TR_new_setter['TR1_outcome'] =np.append(TR_new_setter['TR1_outcome'], full_TR3['TR1_outcome'])
	TR_new_setter['TR2_outcome'] =np.append(TR_new_setter['TR2_outcome'], full_TR3['TR2_outcome'])
	TR_new_setter['TR_outcome'] = np.append(TR_new_setter['TR1_outcome'], TR_new_setter['TR2_outcome'])
	
	
	
	start_time_cons = time.time()
	holder = keep_expanding_mean_iteration.one_iteration(TR_new_setter, labels, binary, ensemble_methods, weight, lb, True, hamming, best_strings_first)
	
	
	start_time_cons = time.time()
	best_classifiers1 = holder['best classifiers']
	TS = holder['TR_set_used']['TS']
	
	prediction_holder1 = predict_prob_classifiers_census(best_classifiers1, binary, best_strings_second, np.array(TS))
	
	for classifier_pred in prediction_holder1:
		classifier_outcomes.append({'type': classifier_pred['type'], 'prediction': classifier_pred['prediction']})


		
	sorted_labprob = np.zeros((len(TS),len(labels)))
	
	#stores the answer for each test sample.	
	answer = np.zeros(len(TS))
	
	counter = 0
	for classifier_prediction in prediction_holder1:
		type = classifier_prediction['type']
		norm_prob = float(best_classifiers[counter]['coefficent']/float(normalization_factor))
		'''
		adds the probabilities of what is in sorted_labprob with predictions
		for ith classifier. 
		'''
		each_ts( classifier_prediction["prob"], sorted_labprob, norm_prob)
		counter = counter +1
		
	for x in range(0, len(sorted_labprob)):
		index = np.argmax(sorted_labprob[x])
		answer[x]= answer[x]+ labels[index]
	
	
	'''
		Finds the class with the largest probability for each test sample
		and stores it in answer.
	'''

	classifier_outcomes.append({"type": "weighted majority voting ensemble", 'prediction': answer, "coefficient": None})
	return classifier_outcomes	
	
	

	
'''
	Adds the probabilities of what is in sorted_labprob with predictions
	for pred. For each additon of pred and sorted_labprob 
	pred is first multipied by norm_prob as it acts as wieght
	for the classifier. 
	
	Returns sorted_labprob that has added the probabilities of pred*weight.
'''
def each_ts(pred, sorted_labprob, norm_prob):
	counter = 0
	for x in range(0, len(pred)):		
		sorted_labprob[counter]= np.add(sorted_labprob[counter], (pred[x]*norm_prob))
		counter = counter+1
		

		
