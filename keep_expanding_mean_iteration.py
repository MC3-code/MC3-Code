from sklearn.grid_search import GridSearchCV
import numpy as np
import data_config
import copy
import new_classifiers5
import itertools
from scipy.spatial.distance import cdist
from sklearn.metrics import normalized_mutual_info_score
	

def remove(hold_expanded, classifiers_length):
	hold_expanded = np.delete(hold_expanded, np.s_[len(hold_expanded[0])-2*classifiers_length:len(hold_expanded[0])-classifiers_length], axis=1)
	return hold_expanded

def create_expanded_sets(classifiers_fitted_TR1, TR2, best_strings):
	predictions_TR2 = copy.deepcopy(TR2)

	for each_classifier in classifiers_fitted_TR1:
		if each_classifier['type'] in best_strings:
			predictions_TR2 = np.column_stack((predictions_TR2, each_classifier['model'].predict(TR2)))	
	return predictions_TR2


def tune_classifier_expanded(TR2_expanded, TR2_outcome,  type, tuned_parameters, model_type):
	if not tuned_parameters:
		model = type
		model.fit(TR2_expanded, TR2_outcome)
	else:
		model = GridSearchCV(type, tuned_parameters, cv=5).fit(TR2_expanded, TR2_outcome)
	return {'model': model, 'type': model_type}



def fit_expanded_set_all_models(TR2_expanded, TR2_outcome, ensemble_methods):
	fitted_models = []
	for each_classifier in new_classifiers5.create_best_classifiers(new_classifiers5.names_all_classifiers(ensemble_methods), ensemble_methods):
		fitted_models.append(tune_classifier_expanded(TR2_expanded, TR2_outcome,  each_classifier['model'], each_classifier['tuned_parameters'], each_classifier['type']))
	return fitted_models
				

def get_predictions(classifiers_fitted_TR2, TS):
	hold_predictions = []
	for each_classifier in classifiers_fitted_TR2:
		if(len(hold_predictions) ==0):
			hold_predictions = each_classifier['model'].predict(TS)
		else:
			hold_predictions = np.column_stack((hold_predictions, each_classifier['model'].predict(TS)))	
	return hold_predictions

def create_all_expanded_sets(TR_set_used, best_strings, ensemble_methods, binary, weight, lb, labels, first_time, TR_models_used):
	#Fit all models on TR1 and TR1 outcome as well as TR2 and TR2_outcome
	if(first_time):
		TR_models_used = {'TR1_model': fit_expanded_set_all_models(TR_set_used['TR1'], TR_set_used['TR1_outcome'], ensemble_methods), 
			'TR2_model': fit_expanded_set_all_models(TR_set_used['TR2'], TR_set_used['TR2_outcome'], ensemble_methods), 
			'TR_model': fit_expanded_set_all_models(TR_set_used['TR'], TR_set_used['TR_outcome'], ensemble_methods)}
	
	
	#Expand TR1 with the predictions of models fitted on TR2 and expand TR2 with the predictions of models fitted on TR1
	#These will be the new "TR1" and new "TR2"
	TR1_expanded = create_expanded_sets(TR_models_used['TR2_model'], TR_set_used['TR1'], best_strings)
	TR2_expanded = create_expanded_sets(TR_models_used['TR1_model'], TR_set_used['TR2'], best_strings)
	TR3_expanded = create_expanded_sets(TR_models_used['TR1_model'], TR_set_used['TR3'], best_strings)
	
	
	TR_set_used['TR1'] = TR1_expanded
	TR_set_used['TR2'] = TR2_expanded
	TR_set_used['TR3'] = TR3_expanded
	
	#Expand TS_expanded_with_TR1 with the predictions of models fitted on TR2
	#Remove the predictions made by the models fitted on TR1 on TS
	#This will be the new "TS"
	
	TR_models_used['TR2_model'] = fit_expanded_set_all_models(TR_set_used['TR2'], TR_set_used['TR2_outcome'], ensemble_methods)
	
	
	TR_models_used['TR1_model'] = fit_expanded_set_all_models(TR_set_used['TR1'], TR_set_used['TR1_outcome'], ensemble_methods)
	

	TS_expanded_with_TR = create_expanded_sets(TR_models_used['TR_model'], TR_set_used['TS'], best_strings)

	
	TR_set_used['TR'] = np.concatenate((TR1_expanded, TR2_expanded), axis=0)
	
	TR_models_used['TR_model'] = fit_expanded_set_all_models(TR_set_used['TR'], TR_set_used['TR_outcome'], ensemble_methods)
	
	TS_expanded_with_TR_expanded = create_expanded_sets(TR_models_used['TR_model'], TS_expanded_with_TR, best_strings)
	TR_set_used['TS'] = remove(TS_expanded_with_TR_expanded, len(best_strings))
	
	second_predictions = get_predictions(TR_models_used['TR_model'], TR_set_used['TS'])
	
	
	return (TR_models_used, second_predictions)
	
	
def distance_between(matrix1, hamming, len_best_strings):
	full_sum = 0.0
        list2 = []
        for i in range(0, len_best_strings):
            list2.append(i)

	largest = 0
	for subset in itertools.combinations(list2, 2):
		nmatrix1 = np.array(matrix1)
		first = [matrix1[:,subset[0]]]
		second = [matrix1[:,subset[1]]]
		if(hamming):
			full_sum = full_sum + cdist(first, second, 'hamming')
			if(cdist(first, second, 'hamming')>largest):
				largest = cdist(first, second, 'hamming')
		else:
			first = np.array(first)
			first = first.flatten()
			
			second = np.array(second)
			second = second.flatten()
			
			hold_nmi = 1-normalized_mutual_info_score(first, second)
			full_sum = full_sum + (hold_nmi)
			if(hold_nmi>largest):
				largest = hold_nmi
			
	return (full_sum, largest)

	
def one_iteration(TR_set, labels, binary, ensemble_methods, weight, lb, test, hamming, best_strings):
	TR_set_used = copy.deepcopy(TR_set)
	
	first_predictions = None
	second_predictions = None
	
	
	best_TR_models = None
	
	
	prev_distance = float("inf")
	distance = float("inf")
	
	first_time = True
	TR_models_used = None
	
	i=0
	while i<5:
		if(prev_distance < distance):
			best_TR_models = TR_models_used
			break
		TR_hold, second_predictions = create_all_expanded_sets(TR_set_used, best_strings, ensemble_methods, binary, weight, lb, labels, first_time, TR_models_used)
		TR_models_used = TR_hold
		first_time = False
		prev_distance = distance
		holdi = distance_between(second_predictions, hamming, len(best_strings))
		distance = holdi[0]
		i = i+1
		if(i==5):
			best_TR_models = TR_models_used
	if(test):		
		return best_classifiers_choice(best_TR_models['TR_model'], TR_set_used['TS'], TR_set_used['TS_outcome'], TR_set_used, best_strings, binary, weight, lb, labels)
	else:
		return best_classifiers_choice(best_TR_models['TR_model'], TR_set_used['TR3'], TR_set_used['TR3_outcome'], TR_set_used, best_strings, binary, weight, lb, labels)

	
def best_classifiers_choice(TR_models, TS, TS_outcome, TR_set_used, best_strings, binary, weight, lb, labels):
	best_classifiers = []
	for each_model in TR_models:
		coor = None
		if(not(binary)):
				coor = normalized_mutual_info_score(each_model['model'].predict(TS), TS_outcome)
		else:
			TR_coor = np.column_stack((each_model['model'].predict(TS), TS_outcome))
			coor = np.corrcoef(TR_coor.T)
			coor = coor[0][1]
		best_classifiers.append({'model2': each_model['model'], 'type': each_model['type'], "coefficent": coor,'fit_x':TR_set_used['TR'], 'fit_y':TR_set_used['TR_outcome']})
	return {'best classifiers': best_classifiers, 'TR_set_used': TR_set_used, "best strings": best_strings}
	
