The main file used to run the program is regular run.py.

There are 5 command line parameters that must be entered (in this order): location of dataset, binary, skip first line, MC3-R, and MC3-S.

1)Location of dataset- Location of the dataset the algorithims/algorithim will be run on
2)Binary- True if the dataset is binary, if the dataset is multiclass then false
3)Skip first line- True if the first line be skipped (as it contains descriptions of each column) else False
4)MC3-R- True if the MC3-R algorithm should be run on the dataset, else False
5)MC3-S- True if the MC3-S algorithm should be run on the dataset, else False


To reiterate the 4 parameters: binary, skip first line, run MC3-R, and run MC3-S are binary choices

What is a binary choice?
Binary choices: Either "True" (or "TRUE") or False or ("FALSE") should be entered for these parameters.

Example:
	This is an example to run Iris.csv which is contained in the dataset folder.
        What would be typed in the command line: python regular_run.py datasets/iris.csv False True True True
        	This would run both MC3-R and MC3-S algorithms on the iris data set which is multiclass, and the first line is not skipped. 

Returned is the roc-auc-score by the algorithim. 
The dataset you entered in is randomoly but with an even distribtuin split into test and train sets. The roc-auc-score is given on how the algorithim performed on the test set
