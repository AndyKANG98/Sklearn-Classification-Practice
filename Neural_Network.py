import scipy.io as sio
import numpy as np
import time
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
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

def select_hidden_units(X_train, y_train, hidden_units, k):
    """ Determine the number of hidden units in single-hidden-layer using cross validation

    Args: 
        X_train: Traning data for cross validation
        y_train: Traing labels for cross validation
        hidden_units: array of candidate hidden units numbers
        k: number of folders in cross validation
    Returns:
        best_h: the number of hidden_units with the highest accuracy in cross_val
    """
    mean_scores = []
    for h in hidden_units:
        # Construct single-hidden-layer neural networks with each h value
        # n_jobs indicates number of Processors, -1 get use of all processors
        # activation="logistic", learning rate=0.001, tol=1e-5 are tested to be suitable according to the performance
        clf = MLPClassifier(hidden_layer_sizes=(h,), activation="logistic", solver='sgd', 
                        alpha=1e-4, learning_rate_init=0.001, tol=1e-5, random_state=520)
        
        # computing the accuracy for each hidden units number, using k-fold cross-validation
        # n_jobs indicates number of Processors, -1 get used of all processors to accelerate the cross validation
        score = cross_val_score(clf, X_train, y_train, cv=k, n_jobs=-1, scoring="accuracy")
        
        mean_scores.append(score.mean())
        print ("When hidden units number = %i, the mean accuracy is: " % h, "{:%}".format(score.mean()))
    
    # Select the best h with the highest mean accuracy 
    best_h = hidden_units[mean_scores.index(max(mean_scores))]
    return best_h


def main():
    # Load the data
    X_train, y_train, X_test, y_test = load_data()

    startTime = time.time()

    # Give the hidden units number candidates
    hidden_units = [1, 5, 10, 20, 50]
    # Determine the best h value using 5-fold cross validation
    best_h = select_hidden_units(X_train, y_train, hidden_units, k=5)

    # Construct single-hidden-layer neural networks with each h value
    # n_jobs indicates number of Processors, -1 get use of all processors
    # activation="logistic", learning rate=0.001, tol=1e-5 are tested to be suitable according to the performance
    clf = MLPClassifier(hidden_layer_sizes=(best_h,), activation="logistic", solver='sgd', 
                     alpha=1e-4, learning_rate_init=0.001, tol=1e-5, random_state=520)

    # Train the classifier and make prediction on testing data
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Print the evaluation results below
    print("The best h is: ", best_h)
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