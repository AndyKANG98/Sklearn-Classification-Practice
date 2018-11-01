import scipy.io as sio
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

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

def main():
    # Load the data
    X_train, y_train, X_test, y_test = load_data()

    startTime = time.time()

    # Construct the classifier with 50-estimator
    # n_jobs indicates number of Processors, -1 get use of all processors
    # "520" as a random state to make the performance stable each time
    clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=520)

    # Train the classifier and make prediction on testing data
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # Print the evaluation results below
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

if __name__ == "__main__":
    main()
