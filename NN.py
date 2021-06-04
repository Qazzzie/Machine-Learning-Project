import utilities
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score

if __name__ == '__main__': #added by kelli to run file in pycharm
    X_train, X_test, Y_train, Y_test = utilities.get_data_as_train_test_split(test_size=.33, random_state=None)
    m = len(Y_train)
    n = len(X_train[0]) + 1
    print(f'Number of training instances is', m)
    print(f'Number of features is', n)

    parameters_to_tweak = {
        'hidden_layer_sizes': [(10, ), (20,), (30,), (20, 10), (20, 10, 20)],
        'activation': ['relu', 'logistic'],
        "solver": ['lbfgs', 'sgd'],
    }
    mlp = MLPClassifier(max_iter=10000, alpha=.0001)
    classifier = GridSearchCV(mlp, param_grid=parameters_to_tweak, verbose=2, scoring='accuracy')

    classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    conf_matrix = confusion_matrix(Y_test, Y_pred)

    print(f"Best hyper parameters from the grid search were {classifier.best_params_}")
    print(f"Accuracy of training set is {classifier.best_score_}")
    print(f'Best test accuracy is', accuracy)
    print(f'Best test precision is', precision_score(Y_test, Y_pred, average='weighted'))
    print(f'Best test recall is', recall_score(Y_test, Y_pred, average='weighted'))
    print(f'Confusion matrix:\n', conf_matrix)
