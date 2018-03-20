-----------------------------------------------------
=== ASSIGNMENT 1 CS 6375.501: Machine Learning ===
-----------------------------------------------------
Name: RICHA SINGH 
Email: <rxs173930@utdallas.edu>
NetID: <rxs173930>

Name: ABHIRAJ DARSHANKAR
Email: <ppd170130@utdallas.edu>
NetID: <ppd170130>

*BOTH SCRIPTS TAKE DATA FROM FOLDER 'Dataset'

DIR STRUCTURE SHOULD LOOK LIKE:
--sentiment_analysis.py
--car_eval_RANDOMFOREST.py	
	Dataset:
	|
	--carTrainData.csv
	--catTestData.csv
	--textTrainData.txt
	--textTestData.txt

_____________________________
CAR TRAIN DATA SET 

Requires Python Version: 2.6 and above
Requires Jupyter Notebook Python 3.6.4 anaconda

Libraries Required: matplotlib ,Pandas, sklearn, numpy, seaborn 

-Loading the dataset from the desired directory .

-Exploratory data Analysis to find out which model to use to best fit the datasets

-During EDA we came to know, most of the columns shows very weak correlation with ‘v7’ column . So, plotting these columns with each other and doing analysis may not give productive results.

-Model Selection : 
-Using Logistic Regression to fit the dataset and the value obtained is 68.76%
-As the model isn’t fitting properly ,to make the data fit better we are using Random Forests Classifiers through which we fit the model upto 93.93%.
-Further, to increase the accuracy of the datasets we are using GridSearch to get 	 combination of best parameters and by using only training tests. The accuracy 		 obtained is 97.29%. Also, we see that the model is best evaluating for max_features=6() and n_estimators= 50 (). 

-By plotting the learning curve we see that the model is overfitting as train accuracy is 1 but test accuracy is much less.

-INFERENCE : We are using Random Forest Classifiers to fit the car train datasets.

______________________________
SENTIMENT ANALYSIS

Requires Python Version: 2.6 and above
Requires Jupyter Notebook Python 3.6.4 anaconda

Libraries Required: Pandas, sklearn, numpy, re 

-Loading the dataset from the desired directory.

-The model prints Confusion matrix, Accuracy,Recall,Precision and F1Score for both test and train datasets. 