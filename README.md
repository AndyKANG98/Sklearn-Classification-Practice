# Sklearn-Classification-Practice
> In this assignment, three classification models are implemented to predict labels for 28*28 grayscale images. The dataset consists of a training set of 10,000 examples and a test set of 1,000 examples, with the corresponding labels. After loading, we have: 

```python
#X_train: training data with shape of (10000, 784)
#y_train: training labels with shape of (10000,)
#X_test: testing data with shape of (1000, 784)
#y_test: testing labels with shape of (1000,)
```

<br>

## **Random Forest**

**Parameter Settings:** 

A Random Forest Classifier is constructed as below: 
```python
# Construct the classifier with 50-estimator
# n_jobs indicates number of Processors, -1 get use of all processors
# "520" as a random state to make the performance stable each time
clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=520)
```
<br>

Some parameters setting experiments below: 
```
n_estimators=5 - Training Score:  0.9831; Test Score:  0.794
n_estimators=10 - Training Score:  0.9953; Test Score:  0.828
n_estimators=20 - Training Score:  0.9994; Test Score:  0.86
n_estimators=50 - Training Score:  1.0; Test Score:  0.862
n_estimators=100 - Training Score:  1.0; Test Score:  0.861
n_estimators=200 - Training Score:  1.0; Test Score:  0.862
```

So, we set n_estimators equal to 50 because according to the observation, keep increasing n_estimators will not further improve the performance when keep increasing the value.

<br> 

**Classification Accuracy:** 
```
Training accuracy:  100.000000%
Testing accuracy:  86.200000%
```
<br>

**Classification Report & Confusion Matrix (Testing):** 
```
===================Classification Report===================


             precision    recall   f1-score   support

          0       0.84      0.87      0.85       107
          1       0.98      0.96      0.97       105
          2       0.72      0.84      0.77       111
          3       0.82      0.87      0.84        93
          4       0.80      0.78      0.79       115
          5       0.95      0.91      0.93        87
          6       0.73      0.59      0.65        97
          7       0.90      0.92      0.91        95
          8       1.00      0.96      0.98        95
          9       0.93      0.95      0.94        95

avg / total       0.86      0.86      0.86      1000

===================Confusion Matrix===================

[[ 93   0   2   4   0   0   8   0   0   0]
 [  1 101   0   3   0   0   0   0   0   0]
 [  3   0  93   2   9   0   4   0   0   0]
 [  4   1   1  81   3   0   3   0   0   0]
 [  0   0  17   4  90   0   4   0   0   0]
 [  0   0   0   0   0  79   0   7   0   1]
 [ 10   0  15   5  10   0  57   0   0   0]
 [  0   0   0   0   0   2   0  87   0   6]
 [  0   1   1   0   0   0   2   0  91   0]
 [  0   0   0   0   0   2   0   3   0  90]]
```
We can see that the random forest classifier gives an 0.86 of precision and recall in testing data. According to the observation of Confusion matrix, the number “6” is relatively hard to predict (only 57 are correctly recognized).

<br>

**Running Time (Exclude loading data):** 
```
The running time is: 1.667402 seconds.
```

<br>

<br>

## **SVM**

**γ Selection:**
```python
# Determine the best gamma value with 5-fold cross validation
best_gamma = select_gamma(X_train, y_train, gammas, k = 5)
```
<br>
Set k=5, using 5-fold cross validation on training instances. 
```python
cross_val_score(clf, X_train, y_train, cv=k, n_jobs=-1, scoring="accuracy")
```
<br>

We get the mean accuracies for different gammas below:
```
When gamma = 0.001000, the mean accuracy is:  10.269990%
When gamma = 0.010000, the mean accuracy is:  10.269990%
When gamma = 0.100000, the mean accuracy is:  10.269990%
When gamma = 1.000000, the mean accuracy is:  10.269990%

The best gamma is:  0.001
```

We see no difference in accuracy regards different gamma. And the performance is all very bad. So whichever gamma selected, it gives the same performance in cross validation.

<br>

**Parameters Settings:** 

An SVM Classifier is constructed as below: 
```python
# Construct SVM classification model with RBF kernel, given the best gamma
clf = SVC(kernel='rbf', gamma=best_gamma, random_state=520)
```

According to some experiments, there’s no improvement by changing some other parameters. And all results are not well on the testing data. So, we just use the default value to construct the SVM Classifier.

<br>

**Classification Accuracy:** 
```
Training accuracy:  100.000000%
Testing accuracy:  10.500000%
```

We can observe that the model is overfitting with very high training accuracy and low testing accuracy.

<br>

**Classification Report & Confusion Matrix (Testing):**
```
===================Classification Report===================

             precision    recall   f1-score   support

          0       0.00      0.00      0.00       107
          1       0.10      1.00      0.19       105
          2       0.00      0.00      0.00       111
          3       0.00      0.00      0.00        93
          4       0.00      0.00      0.00       115
          5       0.00      0.00      0.00        87
          6       0.00      0.00      0.00        97
          7       0.00      0.00      0.00        95
          8       0.00      0.00      0.00        95
          9       0.00      0.00      0.00        95


avg / total       0.01      0.10      0.02      1000


===================Confusion Matrix===================

[[  0 107   0   0   0   0   0   0   0   0]
 [  0 105   0   0   0   0   0   0   0   0]
 [  0 111   0   0   0   0   0   0   0   0]
 [  0  93   0   0   0   0   0   0   0   0]
 [  0 115   0   0   0   0   0   0   0   0]
 [  0  87   0   0   0   0   0   0   0   0]
 [  0  97   0   0   0   0   0   0   0   0]
 [  0  95   0   0   0   0   0   0   0   0]
 [  0  95   0   0   0   0   0   0   0   0]
 [  0  95   0   0   0   0   0   0   0   0]]
```

We can see that all data are predicted as “1”. So, we have a very bad performance with this SVM model.

<br>

**Running Time (Exclude loading data):** 
```
The running time is: 2006.609518 seconds.
```

<br>

<br>

## **Neural Network**


**Number of Hidden Units (H) Selection:**
```python
# Determine the best h value using 5-fold cross validation
best_h = select_hidden_units(X_train, y_train, hidden_units, k=5)
```

Set k=5, using 5-fold cross validation on training instances. 
```python
cross_val_score(clf, X_train, y_train, cv=k, n_jobs=-1, scoring="accuracy")
```

We get the mean accuracies for different number of hidden units below:
```
When hidden units number = 1, the mean accuracy is:  10.189999%
When hidden units number = 5, the mean accuracy is:  44.031475%
When hidden units number = 10, the mean accuracy is:  63.178377%
When hidden units number = 20, the mean accuracy is:  73.607483%
When hidden units number = 50, the mean accuracy is:  80.668708%

The best h is:  50
```
We can see that the mean accuracy increases as the number of hidden units increasing. So, we select the largest value “50” among all the candidates. 

<br>

**Parameters Settings:** 
```python
# Construct single-hidden-layer neural networks with each h value
# n_jobs indicates number of Processors, -1 get use of all processors
clf = MLPClassifier(hidden_layer_sizes=(best_h,), activation="logistic", solver='sgd', alpha=1e-4, learning_rate_init=0.001, tol=1e-5, random_state=520)
```
We set activation="logistic", learning rate=0.001, which are tested to be suitable according to the performance. Some of the parameters experiments below: 
```
When the activation function is logistic (keep learning rate=0.001):  
Training accuracy: 85.200000%; Testing accuracy: 81.600000%
When the activation function is relu (keep learning rate=0.001):  
Training accuracy: 23.300000%; Testing accuracy: 21.300000%
```
```
When learning rate = 0.01 (activation = “logistic”): 
Training accuracy: 70.880000%; Testing accuracy: 68.300000%
When learning rate = 0.001 (activation = “logistic”):
Training accuracy: 85.200000%; Testing accuracy: 81.600000%
When learning rate = 0.0001: (has not converged with in max_iter = 200)
Training accuracy: 84.050000%; Testing accuracy: 78.600000% 
```

<br>

**Classification Accuracy:** 
```
Training accuracy:  85.200000%
Testing accuracy:  81.600000%
```
<br>

**Classification Report & Confusion Matrix (Testing):**
```
===================Classification Report===================

             precision    recall   f1-score   support

          0       0.79      0.84      0.81       107
          1       0.94      0.96      0.95       105
          2       0.66      0.82      0.73       111
          3       0.77      0.81      0.79        93
          4       0.80      0.60      0.69       115
          5       0.96      0.84      0.90        87
          6       0.60      0.52      0.56        97
          7       0.89      0.91      0.90        95
          8       0.93      0.96      0.94        95
          9       0.87      0.95      0.90        95

avg / total       0.82      0.82      0.81      1000

 
===================Confusion Matrix===================

[[ 90   1   2   7   0   0   5   0   2   0]
 [  0 101   0   4   0   0   0   0   0   0]
 [  2   1  91   2   7   0   7   0   1   0]
 [  5   3   1  75   0   0   8   0   1   0]
 [  0   0  28   7  69   0  10   0   1   0]
 [  0   0   0   0   0  73   0   6   0   8]
 [ 17   0  15   3  10   0  50   0   2   0]
 [  0   0   0   0   0   3   0  86   0   6]
 [  0   1   0   0   0   0   3   0  91   0]
 [  0   0   0   0   0   0   0   5   0  90]]

```

After training, the single-hidden-layer neural network with 50 hidden units can get the precision and recall at around 0.82. Same as the Random Forest, the number “6” is relatively hard to predict.

<br>

**Running Time (Exclude loading data):** 
```
The running time is: 196.139197 seconds.
```
