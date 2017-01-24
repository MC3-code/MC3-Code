import csv
import numpy as np
from collections import Counter
import math 
import data_config
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.metrics  import f1_score, mean_squared_error, accuracy_score, confusion_matrix, precision_score, recall_score
import measures
import pandas as pd
from sklearn.feature_extraction import DictVectorizer


def convert_file (filename):
	hold =  pd.DataFrame.from_csv(filename)
	hold = hold.dropna()
	keeper = hold.T.to_dict().values()
	v = DictVectorizer(sparse=False)
	X = v.fit_transform(keeper)
	df = pd.DataFrame(X, columns=v.get_feature_names())
        name = 'binary_'+ filename
	df.to_csv("tester.csv")
