from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn import datasets, linear_model, cross_validation, svm
from sklearn.metrics import accuracy_score, roc_auc_score
import data_config
from sklearn.calibration import CalibratedClassifierCV
import copy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics  import f1_score
'''
	errors to take care of:
		
'''
def multi_class_roc(weight, lb, prediction, TS_outcome, labels):
	prediction_new =np.array(lb.transform(prediction))
	TS_outcome_new = np.array(lb.transform(TS_outcome))
	total_roc_score = 0.0
	
	for each_column in range(0, len(TS_outcome_new[0])):
		roc_score = float(roc_auc_score(TS_outcome_new[:,each_column], prediction_new[:,each_column], average= None))
		total_roc_score =  total_roc_score + float(roc_score*weight[labels[each_column]])
	return total_roc_score
	

def get_new_training_and_set(full_training_set):
	TR, TS = data_config.create_balanced_sets(full_training_set, 0.2)
	
	TR_outcome = TR[:,len(TR[0])-1]
	TR = TR[:,:len(TR[0])-1]
	
	TS_outcome = TS[:,len(TS[0])-1]
	TS = TS[:,:len(TS[0])-1]

	TR_complete = full_training_set[:,len(full_training_set[0])-1]
	TR_complete_outcome = full_training_set[:,:len(full_training_set[0])-1]
	
	return {'TR': TR, 'TR_outcome': TR_outcome, 'TS': TS, 'TS_outcome': TS_outcome}
	
def get_new_training_into3(full_training_set):
	TR, TR3, none = data_config.create_balanced_sets(full_training_set, 0.2)
	TR1, TR2, TR_new = data_config.create_balanced_sets(TR, 0.5)
	
	TR_outcome = TR[:,len(TR[0])-1]
	TR = TR[:,:len(TR[0])-1]
	
	TR_outcome_new = TR_new[:,len(TR_new[0])-1]
	TR_new = TR_new[:,:len(TR_new[0])-1]
	
	TR1_outcome = TR1[:,len(TR1[0])-1]
	TR1 = TR1[:,:len(TR1[0])-1]
	
	TR2_outcome = TR2[:,len(TR2[0])-1]
	TR2 = TR2[:,:len(TR2[0])-1]
	
	TR3_outcome = TR3[:,len(TR3[0])-1]
	TR3 = TR3[:,:len(TR3[0])-1]
	
	TR_complete_outcome = full_training_set[:,len(full_training_set[0])-1]
	TR_complete = full_training_set[:,:len(full_training_set[0])-1]
	
	return {'TR1': TR1, 'TR1_outcome': TR1_outcome, 'TR2': TR2, 'TR2_outcome': TR2_outcome, 'TR3': TR3, 'TR3_outcome': TR3_outcome,
			'TR': TR_new, 'TR_outcome': TR_outcome_new, 'TR_full': TR, "TR_full_outcome": TR_outcome, "TR_complete": TR_complete, "TR_complete_outcome": TR_complete_outcome}
	
def get_new_training_into3_specify(full_training_set, TR3_number, TR2_number):
	TR, TR3, none = data_config.create_balanced_sets(full_training_set, TR3_number)
	TR1, TR2, TR_new = data_config.create_balanced_sets(TR, TR2_number)
	
	TR_outcome = TR[:,len(TR[0])-1]
	TR = TR[:,:len(TR[0])-1]
	
	TR_outcome_new = TR_new[:,len(TR_new[0])-1]
	TR_new = TR_new[:,:len(TR_new[0])-1]
	
	TR1_outcome = TR1[:,len(TR1[0])-1]
	TR1 = TR1[:,:len(TR1[0])-1]
	
	TR2_outcome = TR2[:,len(TR2[0])-1]
	TR2 = TR2[:,:len(TR2[0])-1]
	
	TR3_outcome = TR3[:,len(TR3[0])-1]
	TR3 = TR3[:,:len(TR3[0])-1]
	
	TR_complete = full_training_set[:,len(full_training_set[0])-1]
	TR_complete_outcome = full_training_set[:,:len(full_training_set[0])-1]
	
	return {'TR1': TR1, 'TR1_outcome': TR1_outcome, 'TR2': TR2, 'TR2_outcome': TR2_outcome, 'TR3': TR3, 'TR3_outcome': TR3_outcome,
			'TR': TR_new, 'TR_outcome': TR_outcome_new, 'TR_full': TR, "TR_full_outcome": TR_outcome, "TR_complete": TR_complete, "TR_complete_outcome": TR_complete_outcome }

'''
Given the full training set randomly splits the dataset into TR1 and TR2.
Returns TR1, TR1_outcome and TR2 and TR2_outcome in a dictionary.
'''
def get_new_training(full_training_set):
	TR1, TR2, TR_new = data_config.create_balanced_sets(full_training_set, 0.5)
	
	TR_outcome_new = TR_new[:,len(TR_new[0])-1]
	TR_new = TR_new[:,:len(TR_new[0])-1]
	
	TR1_outcome = TR1[:,len(TR1[0])-1]
	TR1 = TR1[:,:len(TR1[0])-1]
	
	TR2_outcome = TR2[:,len(TR2[0])-1]
	TR2 = TR2[:,:len(TR2[0])-1]
	
	return {'TR1': TR1, 'TR1_outcome': TR1_outcome, 'TR2': TR2, 'TR2_outcome': TR2_outcome, 'TR': TR_new, 'TR_outcome': TR_outcome_new}


def get_specify_training(full_training_set, number):
	TR1, TR2, TR_new = data_config.create_balanced_sets(full_training_set, number)
	
	TR1_outcome = TR1[:,len(TR1[0])-1]
	TR1 = TR1[:,:len(TR1[0])-1]
	
	TR2_outcome = TR2[:,len(TR2[0])-1]
	TR2 = TR2[:,:len(TR2[0])-1]
	
	return {'TR1': TR1, 'TR1_outcome': TR1_outcome, 'TR2': TR2, 'TR2_outcome': TR2_outcome}
'''
Returns a list of tuned parameters for KNN, logistic Regression, SVM, Decision Tree, and Random Forest'
'''
def tuned_parameters():
	tuned_parameters_knn = [{'algorithm': ['ball_tree', 'kd_tree'], 'n_neighbors': [1,3,5]}]
	
	tuned_parameters_logistic = [{'penalty': ['l1', 'l2'], 'C': [0.01, 0.1,1,5,10]}]
	
	tuned_parameters_svm = [{'loss': ['hinge', 'squared_hinge'], 'C': [0.01, 0.1,1,5,10]}]
	
	tuned_parameters_decision = [{'criterion':['gini', 'entropy']}]
	
	tuned_parameters_lda = [{'solver': ['svd', 'lsqr']}]
	
	tuned_parameters_sdg = [{'loss': ['log', 'modified_huber'], 'penalty': ['none', 'l2', 'l1', 'elasticnet']}]
	
	tuned_parameters_randomforest = [{'n_estimators':[1,5,10,25,100]}]
	
	return [tuned_parameters_knn, tuned_parameters_logistic, tuned_parameters_svm, tuned_parameters_decision,
		tuned_parameters_lda, tuned_parameters_sdg, tuned_parameters_randomforest]

			
	
def create_best_classifiers(best_string_classifiers, ensemble_methods):
	keeper = []
	holder = create_classifiers(ensemble_methods)
	
	for each_type in best_string_classifiers:
		keeper.append(finds_match_TR(holder, each_type))
		
	return keeper
	
'''
Returns list of classifiers.
Each classsifier is a dict with a model, tuned_parameters, and type as keys.
'''	
def create_classifiers(ensemble_methods):	
	all_classifiers= tuned_parameters()
	if(ensemble_methods):
		return [{ 'model' : KNeighborsClassifier(), 'tuned_parameters': all_classifiers[0], 'type': "knn"}, 
			{'model' : LogisticRegression(), 'tuned_parameters': all_classifiers[1], 'type': 'Logistic Regression' }, 
			{'model' : svm.LinearSVC(), 'tuned_parameters': all_classifiers[2], 'type': 'SVM' },
			{'model': tree.DecisionTreeClassifier(), 'tuned_parameters': all_classifiers[3], 'type': "Decision Tree"}, 
			{'model': LinearDiscriminantAnalysis(), 'tuned_parameters': all_classifiers[4], 'type': "LDA" },
			{'model': GaussianNB(), 'tuned_parameters': [], 'type': "Naive Bayes"},
			{'model': SGDClassifier(), 'tuned_parameters': all_classifiers[5], 'type': "SDG"},
			{'model': RandomForestClassifier(), 'tuned_parameters': all_classifiers[6], 'type': "Random Forest"}]
	
	else:	
		return [{ 'model' : KNeighborsClassifier(), 'tuned_parameters': all_classifiers[0], 'type': "knn"}, 
			{'model' : LogisticRegression(), 'tuned_parameters': all_classifiers[1], 'type': 'Logistic Regression' }, 
			{'model' : svm.LinearSVC(), 'tuned_parameters': all_classifiers[2], 'type': 'SVM' },
			{'model': tree.DecisionTreeClassifier(), 'tuned_parameters': all_classifiers[3], 'type': "Decision Tree"}, 
			{'model': LinearDiscriminantAnalysis(), 'tuned_parameters': all_classifiers[4], 'type': "LDA" },
			{'model': GaussianNB(), 'tuned_parameters': [], 'type': "Naive Bayes"},
			{'model': SGDClassifier(), 'tuned_parameters': all_classifiers[5], 'type': "SDG"},
			{'model': RandomForestClassifier(), 'tuned_parameters': all_classifiers[6], 'type': "Random Forest"}]		
		

def names_all_classifiers(ensemble_methods):
	if(ensemble_methods):
		return ('knn', 'Logistic Regression', 'SVM', "Decision Tree", "LDA", "Naive Bayes", "SDG", "Random Forest" )

	else:
		return ('knn', 'Logistic Regression', 'SVM', "Decision Tree", "LDA", "Naive Bayes", "SDG", "Random Forest")
		
'''
Given the type of model, training set (TR1 and TR1_outcome), tuned paramters, 
and the set we will be predicting (TR2), and string of model type.
Fits the hyperparamters of the model.
Returns the model and its prediction on TR2, and the type of model.
'''
def tuned_classifier(TR1, TR1_outcome, TR2, type, tuned_parameters, model_type, ensemble_methods):
	prediction = None
	model = None
	#Step 2- K-fold cross validation on TR1 to obtain optimal regression model, model. 
	if not tuned_parameters:
		model = type
		model.fit(TR1, TR1_outcome)
		if(ensemble_methods):
			model = BaggingClassifier(model, max_samples=1.0, max_features=1.0).fit(TR1, TR1_outcome)
		prediction = model.predict_proba(TR2)
	else:
		model = GridSearchCV(type, tuned_parameters, cv=10, scoring="accuracy").fit(TR1, TR1_outcome)
		if(ensemble_methods):
			model = BaggingClassifier(model.best_estimator_, max_samples=1.0, max_features=1.0).fit(TR1, TR1_outcome)
		if(model_type=="SVM" or model_type=="Decision Tree"  or model_type=="SDG"):
			clf_isotonic = CalibratedClassifierCV(model, cv='prefit', method='sigmoid').fit(TR1, TR1_outcome)
			prediction = np.array(clf_isotonic.predict_proba(TR2))
		else:
			prediction = np.array(model.predict_proba(TR2))
		
	return { 'prediction': prediction, 'model': model, 'type': model_type}
	
	
'''
Given the training set (TR1 and TR1_outcome), and the training set to be predicted (TR2).
Returns an aggregrated prediction of each classifier prediction and TR2, and returns
each model after it has been optimized and fitted on TR1 and TR1_outcome. 
'''


def create_expanded_sets(TR1, TR1_outcome, TR2, TR2_outcome, new_features_only, best_strings, ensemble_methods):
	#Stores TR2_expanded.
	predictions_TR2 = []
	if(not(new_features_only)):
		predictions_TR2 = TR2
	#Stores each clasiffier, and name
	classifier_store = []
	for each_classifier in create_best_classifiers(best_strings, ensemble_methods):
		tuned_holder = tuned_classifier(TR1, TR1_outcome, TR2, each_classifier['model'], 
			each_classifier['tuned_parameters'], each_classifier['type'], ensemble_methods)
			
		if(len(predictions_TR2)<1):
			predictions_TR2 = tuned_holder['prediction']
		else:
			predictions_TR2 = np.column_stack((predictions_TR2, tuned_holder['prediction']))
		
		classifier_store.append({'model': tuned_holder['model'], 'type': tuned_holder['type']})
		
	return {"TR2_pred" : predictions_TR2, "stored_classifiers": classifier_store}
	
'''
Given the expanded data set TR2_expanded (aggregated with the predictions of the classifiers), 
its outcome TR2_outcome, the type of model, and its name, model type.
Returns a dict of fitted model and optimized model on TR2_expanded, mean_score when fitted, and	
the type of model. 
'''
def tune_classifier_expanded(TR2_expanded, TR2_outcome,  type, tuned_parameters, model_type, ensemble_methods):
	model1 = None
	if not tuned_parameters:
		model = type
		model.fit(TR2_expanded, TR2_outcome)
		score = float(model.score(TR2_expanded, TR2_outcome))
		if(ensemble_methods):
			model1 = BaggingClassifier(model, max_samples=1.0, max_features=1.0).fit(TR2_expanded, TR2_outcome)
		else:
			model1 = model
	else:
		model = GridSearchCV(type, tuned_parameters, cv=10).fit(TR2_expanded, TR2_outcome)
		if(ensemble_methods):
			model1 = BaggingClassifier(model.best_estimator_, max_samples=1.0, max_features=1.0).fit(TR2_expanded, TR2_outcome)
		else:
			model1 = model
	return {'model_bagged': model1, 'type': model_type, 'model': model }
	
'''
Given a full training set.
Returns the score of model2, model1, model2, and the type of classifier.	
'''	
def one_iteration(TR_set, training_set3, new_features_only, labels, binary, best_strings_first, best_strings_second, ensemble_methods, lb, weight):
	store_TR = []
	
	#store_expanded_sets contains TR2_expanded and the models used to do this.
	store_expanded_sets = create_expanded_sets(TR_set['TR1'], TR_set['TR1_outcome'], TR_set['TR2'], TR_set['TR2_outcome'], new_features_only, best_strings_first, ensemble_methods)
	
	for each_classifier in create_best_classifiers(best_strings_second, ensemble_methods):
		store = {}
		
		store = tune_classifier_expanded(store_expanded_sets['TR2_pred'], TR_set['TR2_outcome'],  each_classifier['model'], 
				each_classifier['tuned_parameters'], each_classifier['type'], ensemble_methods)		
		
		hold = { "model2" : store['model_bagged'], "model_unbagged": store['model'] ,"type" : each_classifier['type'],
			'fit_x':store_expanded_sets['TR2_pred'], 'fit_y':TR_set['TR2_outcome']}
		
		store_TR.append(hold)
	
	return store_TR
	
	
	
def finds_match_TR(store_TR, type):
	for each_classifier in store_TR:
		if each_classifier['type'] == type:
			return each_classifier	


	
