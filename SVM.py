import scipy.io as sio
import numpy as np
import time
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


def load_data():
    """ Load the .mat files from the Dataset

    Returns: 
        X_train: training data with shape of (10000, 784)
        y_train: training labels with shape of (10000,)
        X_test: testing data with shape of (1000, 784)
        y_test: testing labels with shape of (1000,)
    """
    X_train = sio.loadmat("./Dataset/train_images")["train_images"]
    y_train = sio.loadmat("./Dataset/train_labels")["train_labels"]
    X_test = sio.loadmat("./Dataset/test_images")["test_images"]
    y_test = sio.loadmat("./Dataset/test_labels")["test_labels"]

    y_train = np.reshape(y_train, -1)
    y_test = np.reshape(y_test, -1)

    return X_train, y_train, X_test, y_test

def scale_data(X):
    """ Preprocess the X data with StandardScaler

    Returns: 
        X_scale: Standardize features by removing the mean and scaling to unit variance
    """
    scaler = StandardScaler().fit(X)

    X_scale = scaler.transform(X)

    return X_scale

def select_gamma(X_train, y_train, gammas, k):
    """ Determine the kernel parameter best_gamma using cross validation

    Args: 
        X_train: Traning data for cross validation
        y_train: Traing labels for cross validation
        gammas: array of candidate gamma values
        k: number of folders in cross validation
    Returns:
        best_gamma: the gamma value with the highest accuracy in cross_val
    """
    mean_scores = []
    for g in gammas:
        # Construct SVM classification model with RBF kernel, and each value in gammas
        # "520" as a random state to make the performance stable each time
        clf = SVC(kernel='rbf', gamma=g, random_state=520)
        
        # computing the accuracy for each gamma, using k-fold cross-validation
        # n_jobs indicates number of Processors, -1 get use of all processors to accelerate the cross validation
        score = cross_val_score(clf, X_train, y_train, cv=k, n_jobs=-1, scoring="accuracy")

        mean_scores.append(score.mean())
        print ("When gamma = %f, the mean accuracy is: " % g, "{:%}".format(score.mean()))

    # Select the best gamma with the highest mean accuracy
    best_gamma = gammas[mean_scores.index(max(mean_scores))]
    return best_gamma


def main():
    # Load the data (Select to load original data or preprocessed data)
    X_train, y_train, X_test, y_test = load_data()
    
    startTime = time.time()

    # Transform the X data with StandardScaler
    X_train = scale_data(X_train)
    X_test = scale_data(X_test)
    
    # Give the gamma candidate values
    gammas = [0.001, 0.01, 0.1, 1]
    # Determine the best gamma value with 5-fold cross validation
    best_gamma = select_gamma(X_train, y_train, gammas, k = 5)

    # Construct SVM classification model with RBF kernel, given the best gamma
    clf = SVC(kernel='rbf', gamma=best_gamma, random_state=520)
    
    # Train the classifier and make prediction on testing data
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Print the evaluation results below
    print("The best gamma is: ", best_gamma)
    print("Training accuracy: ", "{:%}".format(clf.score(X_train, y_train)))
    print("Testing accuracy: ", "{:%}".format(clf.score(X_test, y_test)))
    print()
    print("===================Classification Report===================")
    print()
    print(classification_report(y_test, y_pred))
    print()
    print("===================Confusion Matrix===================")
    print()
    print(confusion_matrix(y_test, y_pred))
    print()
    print("The running time is: %f seconds." % (time.time() - startTime))


if __name__== "__main__":
    main()