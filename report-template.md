# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### NAME - Mohamad Samruk

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
Kaggle doesn’t accept negative predictions, so all negative predictions must be set to 0 to make successful submission.  

### What was the top ranked model that performed?
In the first training round, The model was WeightedEnsemble_L3.

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features? 
To gain insights into the distribution of the dataset, I plotted histograms for each feature. Additionally, I created a correlation matrix to identify the correlation between variables. Utilizing the Datetime column, I generated additional features such as hour, day, and month. 

### How much better did your model preform after adding additional features and why do you think that is?
The Kaggle score improved from 1.400 to 0.485 due to the incorporation of additional predictive features, which enhanced the models’ predictive capabilities.

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
Despite experimenting with various hyperparameter combinations, the model’s performance exhibited a modest improvement. This suggests that Autogluon’s capabilities in generating optimal hyperparameter choices during the modeling process may have contributed to this outcome. Notably, Autogluon is an AutoML solution, which implies its potential to identify suitable hyperparameters for the given task. 

### If you were given more time with this dataset, where do you think you would spend more time?
Creating additional features would give more performance gain compared to hyperparameter tuning.

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
|model|RF-n_estimators|GBM-max_depth|XGM-learning_rate|score|
|--|--|--|--|--|
|initial|default|default|default|1.400|
|add_features|default|default|default|0.485|
|hpo|95|8|0.01|0.484|

### The line plot showing the top model score for across training run X-axis - training run, Y-axis - model score


![model_train_score.png](/model_train_score.png)

### The line plot showing the top kaggle score for across training run X-axis - training run, Y-axis - Kaggle score 


![model_test_score.png](/model_test_score.png)

## Summary

Predicting bike-sharing demand is a prevalent challenge faced by ride-sharing companies, and it falls under the realm of regression analysis. To address this problem, I employed the AutoML solution Autogluon. Autogluon trains multiple models concurrently and utilizes an ensemble method to enhance the accuracy of predictions. By combining the creation of additional features and hyperparameter tuning, I developed models that enable the accurate prediction of bike-sharing demand for given features with confidence. 
