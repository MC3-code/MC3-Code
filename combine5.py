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
import keep_expanding4
from sklearn.metrics  import f1_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import measures
from sklearn.ensemble import BaggingClassifier
import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

def feature_weighted_linear_stacking(best_classifiers, TR_set, new_features_only, best_strings_first, ensemble_methods):
	TR3_expanded = new_classifiers5.create_expanded_sets(TR_set['TR1'], TR_set['TR1_outcome'], TR_set['TR3'], TR_set['TR3_outcome'],
		new_features_only, best_strings_first, ensemble_methods)["TR2_pred"]
		
	TR3_outcome_neg =copy.deepcopy(TR_set['TR3_outcome'])
	
	TR3_outcome_neg[TR3_outcome_neg == 0] = -1
	
	
	training_sample = []
	for sample in range(0, len(TR3_expanded)):
		holder = []
		for value in best_classifiers:
			 store_predictions = value['model2'].predict([TR3_expanded[sample]])
			 if(store_predictions == 0):
				store_predictions = -1
			 if(len(holder) == 0):
				holder = (TR3_expanded[sample]*store_predictions)
			 else:
				holder = np.append(holder, TR3_expanded[sample]*store_predictions)
		training_sample.append(holder)
		
	hold = LogisticRegression().fit(training_sample, TR3_outcome_neg)
	
	return hold
	
def TS_feature_weighted_linear_stacking(TS, TS_outcome, best_classifiers, TR_set, new_features_only, best_strings_first, ensemble_methods):
	TS_expanded = create_TS_expanded(TR_set['TR'], TR_set['TR_outcome'], TS, TS_outcome, new_features_only, best_strings_first, ensemble_methods)
	
	testing_sample = []	
	for sample in range(0, len(TS_expanded)):
		holder = []
		for value in best_classifiers:
			 store_predictions = value['model2'].predict([TS_expanded[sample]])
			 if(store_predictions == 0):
				store_predictions = -1
			 if(len(holder) == 0):
				holder = (TS_expanded[sample]*store_predictions)
			 else:
				holder = np.append(holder, TS_expanded[sample]*store_predictions)
		testing_sample.append(holder)
	return testing_sample

def combine_baseline(model_dict, TS, labels, TR, TR_outcome):
	
	normalization_factor = 0.0
	
	#stores probability of each class label
	sorted_labprob = np.zeros((len(TS),len(labels)))
	
	#stores the answer for each test sample.	
	answer = np.zeros(len(TS))
	
	measure_diversity = []
	
	
	for classifier_prediction in model_dict:
		normalization_factor = normalization_factor + classifier_prediction['coefficent']
	
	for classifier_prediction in model_dict:
		type = classifier_prediction['type']
		norm_prob = classifier_prediction['coefficent']/normalization_factor
		'''
		adds the probabilities of what is in sorted_labprob with predictions
		for ith classifier. 
		'''
		
		measure_diversity.append((type, classifier_prediction['model'].predict(TS),1))
		
		predictions = []
		if(type =="SVM" or type== "Decision Tree" or type=='SDG'):
			clf_isotonic = CalibratedClassifierCV(classifier_prediction['model'], cv='prefit', method='sigmoid').fit(TR, TR_outcome)
			prediction = np.array(clf_isotonic.predict_proba(TS))
		else:
			predictions = classifier_prediction['model'].predict_proba(TS)
			
		each_ts(predictions, sorted_labprob, norm_prob)

	for x in range(0, len(sorted_labprob)):
		index = np.argmax(sorted_labprob[x])
		answer[x]= answer[x]+ labels[index]
		
	
	#print "baseline diversity"
	#print measures.diversity_measure(measure_diversity, TS, labels)
		
	return answer

def create_TS_expanded(TR, TR_outcome, TS, TS_outcome, new_features_only, best_strings_first, ensemble_methods):
	expanded_sets = new_classifiers5.create_expanded_sets(TR, TR_outcome, TS, TS_outcome, new_features_only, best_strings_first,ensemble_methods )
	return expanded_sets['TR2_pred']
	
	
	
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
				bagging = None
				predict_holder.append({'type': value['type'], "prob": value['model2'].predict_proba(TS_expanded), "prediction": value['model2'].predict(TS_expanded), "coefficent": value["coefficent"], "model": value['model2']})
			'''
			print value['type']
			print "bagged"	
			print roc_auc_score(TS_outcome, value['model2'].predict(TS_expanded), average= None)
			print "not bagged"
			print roc_auc_score(TS_outcome, value['model_unbagged'].predict(TS_expanded), average= None)
			'''
				#if(value['type']=="Naive Bayes"):
					#bagging = BaggingClassifier(GaussianNB(), max_samples=0.2, max_features=1.0, n_estimators=1).fit(value['fit_x'], value['fit_y'])
				#else:
					#bagging = BaggingClassifier(value['model2'].best_estimator_, max_samples=0.2, max_features=1.0, n_estimators=1).fit(value['fit_x'], value['fit_y'])
				
			#if(value['type']!="SVM"):
				#measure_diversity.append((value['type'] , bagging.predict(TS_expanded),1))
			measure_diversity_regular.append((value['type'] , value['model2'].predict(TS_expanded) ,1))
	#print "algorithm diversity"
	#print measures.diversity_measure(measure_diversity, TS, labels)
	#print "without bagging"
	#print measures.diversity_measure(measure_diversity_regular, TS, labels)
	
	'''
	print " "
	print " "
	print " "
	print " "
	print " "
	'''
	
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
	
	TR = TR_set_used["TR_complete"]
	
	TR_outcome = TR_set_used["TR_complete_outcome"]
	
	prediction_holder_TR3 = predict_prob_classifiers(best_classifiers, TR_set_used['TR'], TR_set_used['TR_outcome'], TR_set_used['TR3'], TR_set_used['TR3_outcome'], new_features_only, binary, best_strings_first, best_strings_second, labels, ensemble_methods)
	
	
	classifier_outcomes = []
	
	#stores probability of each class label
	
	
	
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
	
	best_classifiers1 = new_classifiers5.one_iteration(TR_new_setter, False, new_features_only, labels, binary, best_strings_first, best_strings_second, ensemble_methods, weight, lb)
	
	
	prediction_holder1 = predict_prob_classifiers(best_classifiers1, TR, TR_outcome, TS, TS_outcome, new_features_only, binary, best_strings_first, best_strings_second, labels, ensemble_methods)
	
	for classifier_pred in prediction_holder1:
		classifier_outcomes.append({'type': classifier_pred['type'], 'prediction': classifier_pred['prediction']})
		

	hold = feature_weighted_linear_stacking(best_classifiers, TR_set_used, new_features_only, best_strings_first, ensemble_methods)
	
	keeper = TS_feature_weighted_linear_stacking(TS, TS_outcome, best_classifiers, TR_set_used, new_features_only, best_strings_first, ensemble_methods)
	
	classifier_outcomes.append({"type": "feature_weighted_ensemble", 'prediction': hold.predict(keeper), "coefficient": None})

		
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
	#tests.combine_test(prediction_holder, TS_outcome, answer)
	return classifier_outcomes
	
	
	
	
	
	
	
def feature_weighted_linear_stacking_census(best_classifiers, TR_set, new_features_only, best_strings_first, ensemble_methods):
	TR3_expanded = TR_set['TR3']
		
	TR3_outcome_neg =copy.deepcopy(TR_set['TR3_outcome'])
	
	TR3_outcome_neg[TR3_outcome_neg == 0] = -1
	
	
	training_sample = []
	for sample in range(0, len(TR3_expanded)):
		holder = []
		for value in best_classifiers:
			 store_predictions = value['model2'].predict([TR3_expanded[sample]])
			 if(store_predictions == 0):
				store_predictions = -1
			 if(len(holder) == 0):
				holder = (TR3_expanded[sample]*store_predictions)
			 else:
				holder = np.append(holder, TR3_expanded[sample]*store_predictions)
		training_sample.append(holder)
		
	hold = LogisticRegression().fit(training_sample, TR3_outcome_neg)
	
	return hold
	
	
def TS_feature_weighted_linear_stacking_census(TS, TS_outcome, best_classifiers, TR_set, new_features_only, best_strings_first, ensemble_methods):
	TS_expanded = TR_set['TS']
	
	testing_sample = []	
	for sample in range(0, len(TS_expanded)):
		holder = []
		for value in best_classifiers:
			 store_predictions = value['model2'].predict([TS_expanded[sample]])
			 if(store_predictions == 0):
				store_predictions = -1
			 if(len(holder) == 0):
				holder = (TS_expanded[sample]*store_predictions)
			 else:
				holder = np.append(holder, TS_expanded[sample]*store_predictions)
		testing_sample.append(holder)
	return testing_sample
	
	
	
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
	
	TR = full_training_set[:,:len(full_training_set[0])-1]
	
	TR_outcome = full_training_set[:,len(full_training_set[0])-1]
	
	prediction_holder_TR3 = predict_prob_classifiers_census(best_classifiers, binary, best_strings_second, TR_set_used['TR3'])
	
	stack_TR3 = []
	for each_classifier1 in prediction_holder_TR3:
		if(len(stack_TR3)==0):
			stack_TR3 = each_classifier1["prob"]
		else:
			stack_TR3 = np.column_stack((stack_TR3, each_classifier1["prob"]))
	
	
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
	holder = keep_expanding4.one_iteration(TR_new_setter, labels, binary, ensemble_methods, weight, lb, True, hamming)
	best_classifiers1 = holder['best classifiers']
	TS = holder['TR_set_used']['TS']
	
	prediction_holder1 = predict_prob_classifiers_census(best_classifiers1, binary, best_strings_second, np.array(TS))
	
	for classifier_pred in prediction_holder1:
		classifier_outcomes.append({'type': classifier_pred['type'], 'prediction': classifier_pred['prediction']})
		
	
	stack = []
	for each_classifier in prediction_holder1:
		if(len(stack)==0):
			stack = each_classifier["prob"]
		else:
			stack = np.column_stack((stack, each_classifier["prob"]))
	
	
	hold = feature_weighted_linear_stacking_census(best_classifiers, TR_set_used, new_features_only, best_strings_first, ensemble_methods)
	
	keeper = TS_feature_weighted_linear_stacking_census(TS, TS_outcome, best_classifiers, TR_set_used, new_features_only, best_strings_first, ensemble_methods)
	
	classifier_outcomes.append({"type": "feature_weighted_ensemble", 'prediction': hold.predict(keeper), "coefficient": None})

		
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
	#tests.combine_test(prediction_holder, TS_outcome, answer)
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
		

		