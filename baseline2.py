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
import copy
import new_classifiers5
import itertools
import combine5
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics  import f1_score
import time

def baseline(TR_set, labels, ensemble_methods, lb, weight, binary):
	keeper = best_parameters(TR_set, ensemble_methods, binary)
	each_classifier(keeper, TR_set, labels, binary, weight, lb)
	
	linear_stacking(TR_set, ensemble_methods, weight, lb, binary, labels)
	
	random_forest(TR_set, weight, lb, binary, labels)
	bagging(TR_set, keeper, weight, lb, binary, labels)
	ada_boost(TR_set, keeper, weight, lb, binary, labels)
	#gradient_boost(TR_set, weight, lb, binary, labels)

	
def change_to_binary(TR_coor):
	for number in range(0, len(TR_coor)):
		if(not(TR_coor[number][0] == TR_coor[number][1])):
			TR_coor[number][0] = 0
			TR_coor[number][1] = 1
	return TR_coor
			
	
def best_parameters(TR_set, ensemble_methods, binary):
	keeper = []
	for value in new_classifiers5.create_best_classifiers(new_classifiers5.names_all_classifiers(ensemble_methods), ensemble_methods):
		keeper.append(tuned_classifier_TR3(TR_set, value['model'], value['tuned_parameters'], value['type'], binary))
	return keeper

def tuned_classifier_TR3(TR_set, type, tuned_parameters, model_type, binary):
	model = None
	if not tuned_parameters:
		model = type
		model.fit(TR_set['TR'], TR_set['TR_outcome'])
		TR_coor = np.column_stack((model.predict(TR_set['TR3']), TR_set['TR3_outcome']))
		if(not(binary)):
			TR_coor = change_to_binary(TR_coor)
		coor = np.corrcoef(TR_coor.T)
		coor = coor[0][1]
		return {'model': model, 'type': model_type, "coefficent": coor}
	else:
		model = GridSearchCV(type, tuned_parameters, cv=10).fit(TR_set['TR'], TR_set['TR_outcome'])
		TR_coor = np.column_stack((model.best_estimator_.predict(TR_set['TR3']), TR_set['TR3_outcome']))
		if(not(binary)):
			TR_coor = change_to_binary(TR_coor)
		coor = np.corrcoef(TR_coor.T)
		coor = coor[0][1]
		return {'model': model.best_estimator_ , 'type': model_type, "coefficent": coor}
		
		
def tuned_classifier(TR1, TR1_outcome, type, tuned_parameters, model_type):
	model = None
	if not tuned_parameters:
		model = type
		model.fit(TR1, TR1_outcome)
		return {'model': model, 'type': model_type}
	else:
		model = GridSearchCV(type, tuned_parameters, cv=10).fit(TR1, TR1_outcome)
		return {'model': model.best_estimator_ , 'type': model_type}
		
		
	

def create_baseline(TR, TR_outcome, TS, model):
	model.fit(TR, TR_outcome)
	return model.predict(TS)
	

	
def predictive_measures(type_of_score, TS_pred, TS_outcome, average_b, labels, weight, lb):
	if(average_b == "binary"):
		print "   roc_auc_score: " +  str(roc_auc_score(TS_outcome, TS_pred, average= 'weighted'))
		print "      f-score: " +  str(f1_score(TS_outcome, TS_pred, average='binary'))
	else:
		print "   roc_auc_score: " +  str(new_classifiers5.multi_class_roc(weight, lb, TS_pred, TS_outcome, labels))
		print "      f-score: " +  str(f1_score(TS_outcome, TS_pred, average='weighted'))
	
def baseline_score(name, TR, TR_outcome, TS, TS_outcome, baseline_classifiers, binary, labels, weight, lb):
	baseline_model = (new_classifiers5.finds_match_TR(baseline_classifiers,name))
	TS_baseline_pred = create_baseline(TR, TR_outcome, TS, baseline_model['model'])
	if(binary):
		print str(name)
		predictive_measures("baseline", TS_baseline_pred, TS_outcome, "binary", labels, weight, lb)
	else:
		print str(name)
		predictive_measures("baseline", TS_baseline_pred, TS_outcome, "weighted", labels, weight, lb)

def ensemble(keeper, TR_set, labels, weight, lb, binary):
	start_time = time.time()
	answer = combine5.combine_baseline(keeper, TR_set['TS'], labels, TR_set['TR'], TR_set['TR_outcome'])
	print "ensemble"
	if(binary):
		print "   roc_auc_score: " + str(roc_auc_score(TR_set['TS_outcome'], answer, average= 'weighted'))
		print "                 f-score: " +  str(f1_score(TR_set['TS_outcome'], answer, average='binary'))
	else:
		print "   roc_auc_score: " +  str(new_classifiers5.multi_class_roc(weight, lb, answer, TR_set['TS_outcome'], labels))
		print "      f-score: " +  str(f1_score(TR_set['TS_outcome'], answer, average='weighted'))
	print("--- %s seconds ---" % (time.time() - start_time))
	
		
def each_classifier(keeper, TR_set, labels, binary, weight, lb):
	print "Optimized parameters for each classifier on TR + weighted combination in ensemble"
	for value in keeper:
		start_time = time.time()
		baseline_score(value['type'], TR_set['TR'], TR_set['TR_outcome'], TR_set['TS'], TR_set['TS_outcome'], keeper, binary, labels, weight, lb)
		print("--- %s seconds ---" % (time.time() - start_time))
	ensemble(keeper, TR_set, labels, weight, lb, binary)
	print " "
		

		
		
		
		
def random_forest(TR_set, weight, lb, binary, labels):
	start_time = time.time()
	model = GridSearchCV(RandomForestClassifier(), [{'n_estimators':[1,5,10,25,100]}], cv=10).fit(TR_set['TR'], TR_set['TR_outcome'])
	predictions = model.predict(TR_set['TS'])
	print "random forest"
	if(binary):
		print "   roc_auc_score: " + str(roc_auc_score(TR_set['TS_outcome'], predictions, average= None))
		print "                 f-score: " +  str(f1_score(TR_set['TS_outcome'], predictions, average='binary'))
	else:
		print "   roc_auc_score: " +  str(new_classifiers5.multi_class_roc(weight, lb, predictions, TR_set['TS_outcome'], labels))
		print "      f-score: " +  str(f1_score(TR_set['TS_outcome'], predictions, average='weighted'))
	print("--- %s seconds ---" % (time.time() - start_time))
	print " "
	
def bagging(TR_set, keeper, weight, lb, binary, labels):
	print " "
	print "bagging"
	start_time = time.time()
	for value in keeper:
		bagging_each(value['model'], TR_set, value['type'], weight, lb, binary, labels)
	print("--- %s seconds ---" % (time.time() - start_time))
	print " "
	
	
def bagging_each(model, TR_set, type, weight, lb, binary, labels): 
	start_time = time.time()
	hold = BaggingClassifier(model, max_samples=1.0, max_features=1.0).fit(TR_set['TR'], TR_set['TR_outcome'])
	predictions = hold.predict(TR_set['TS'])
	if(binary):
		print str(type) + ":   roc_auc_score: " +str(roc_auc_score(TR_set['TS_outcome'],predictions, average= 'weighted'))
		print "                 f-score: " +  str(f1_score(TR_set['TS_outcome'], predictions, average='binary'))
	else:
		print str(type) + ":   roc_auc_score: " +str(new_classifiers5.multi_class_roc(weight, lb, predictions, TR_set['TS_outcome'], labels))
		print "                 f-score: " +  str(f1_score(TR_set['TS_outcome'], predictions, average='weighted'))
	print("--- %s seconds ---" % (time.time() - start_time))



	
'''
def ada_boost(TR_set, keeper):
	print " "
	print "boosting"
	for value in keeper:
		adaboost_each(value['model'], TR_set, value['type'])
	print " "
	
def adaboost_each(model, TR_set, type):
	tuned_parameters_ada = [{'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,25,50, 75,100]}]
	model = GridSearchCV(AdaBoostClassifier(base_estimator=model), tuned_parameters_ada, cv=5).fit(TR_set['TR'], TR_set['TR_outcome'])
	print "adaboost: " + "   roc_auc_score: " +str(roc_auc_score(TR_set['TS_outcome'], model.predict(TR_set['TS']), average= None))
'''

def ada_boost(TR_set, keeper, weight, lb, binary, labels):
	start_time = time.time()
	print " "
	print "boosting"
	find_decision_tree = new_classifiers5.finds_match_TR(keeper, "Decision Tree")['model']
	tuned_parameters_ada = [{'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,25,50, 75,100]}]
	model = GridSearchCV(AdaBoostClassifier(base_estimator=find_decision_tree), tuned_parameters_ada, cv=10).fit(TR_set['TR'], TR_set['TR_outcome'])
	predictions = model.predict(TR_set['TS'])
	if(binary):
		print "adaboost: " + "   roc_auc_score: " +str(roc_auc_score(TR_set['TS_outcome'], predictions, average= 'weighted'))
		print "                 f-score: " +  str(f1_score(TR_set['TS_outcome'], predictions, average='binary'))
	else:
		print "adaboost: " + "   roc_auc_score: " +str(new_classifiers5.multi_class_roc(weight, lb, predictions, TR_set['TS_outcome'], labels))
		print "                 f-score: " +  str(f1_score(TR_set['TS_outcome'], predictions, average='weighted'))
	print("--- %s seconds ---" % (time.time() - start_time))
	
	
def gradient_boost(TR_set, weight, lb, binary, labels):
	start_time = time.time()
	tuned_parameters_gradient = [{'loss': ['deviance', 'exponential'], 'n_estimators': [1,25,50, 75,100], 'learning_rate':[0.01, 0.05, 0.1, 0.25, 0.5, 1] }]
	model = GridSearchCV(GradientBoostingClassifier(), tuned_parameters_gradient, cv=10).fit(TR_set['TR'], TR_set['TR_outcome'])
	predictions= model.predict(TR_set['TS'])
	if(binary):	
		print "gradientboost: " + "   roc_auc_score: " +str(roc_auc_score(TR_set['TS_outcome'], predictions, average= 'weighted'))
		print "                 f-score: " +  str(f1_score(TR_set['TS_outcome'], predictions, average='binary'))
	else:
		print "gradientboost: " + "   roc_auc_score: " +str(new_classifiers5.multi_class_roc(weight, lb, predictions, TR_set['TS_outcome'], labels))
		print "                 f-score: " +  str(f1_score(TR_set['TS_outcome'], predictions, average='weighted'))
	print " "
	print("--- %s seconds ---" % (time.time() - start_time))
	
	
	

def linear_stacking(TR_set, ensemble_methods, weight, lb, binary, labels):
	start_time = time.time()
	TR2_predictions = []
	TS_predictions = []
	hold_sets =new_classifiers5.get_new_training(np.column_stack((TR_set['TR_full'], TR_set["TR_full_outcome"])))
	for each_classifier in new_classifiers5.create_best_classifiers(new_classifiers5.names_all_classifiers(ensemble_methods), ensemble_methods):
		store = tuned_classifier(hold_sets['TR1'], hold_sets['TR1_outcome'], 
			each_classifier['model'], each_classifier['tuned_parameters'], each_classifier['type'])
			
		if(len(TR2_predictions) ==0):
			TR2_predictions = store['model'].predict(hold_sets['TR2'])
			TS_predictions = store['model'].predict(TR_set['TS'])
		else:
			TR2_predictions = np.column_stack((TR2_predictions, store['model'].predict(hold_sets['TR2'])))
			TS_predictions = np.column_stack((TS_predictions, store['model'].predict(TR_set['TS'])))
	
	tuned_parameters_logistic = [{'penalty': ['l1', 'l2'], 'C': [0.01, 0.1,1,5,10]}]
	model = GridSearchCV(LogisticRegression(), tuned_parameters_logistic, cv=5).fit(TR2_predictions, hold_sets['TR2_outcome'])
	predictions = model.predict(TS_predictions)
	
	if(binary):
		print "linear stacking   roc_auc_score: " + str(roc_auc_score(TR_set['TS_outcome'], predictions, average= 'weighted'))
		print "                  f-score: " +  str(f1_score(TR_set['TS_outcome'], predictions, average='binary'))
	else:
		print " linear stacking  roc_auc_score: " +  str(new_classifiers5.multi_class_roc(weight, lb, predictions, TR_set['TS_outcome'], labels))
		print "      f-score: " +  str(f1_score(TR_set['TS_outcome'], predictions, average='weighted'))
	print("--- %s seconds ---" % (time.time() - start_time))
	print " "
