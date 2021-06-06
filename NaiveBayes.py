from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
import utilities

if __name__ == '__main__':
    #split into train and test sets
    X_train, X_test, Y_train, Y_test = utilities.get_data_as_train_test_split(test_size=.33, random_state=None)

    m = len(Y_train)
    n = len(X_train[0]) + 1
    print(f'Number of training instances is', m)
    print(f'Number of features is', n)

    gnb = GaussianNB()
    gnb.fit(X_train, Y_train)
    Y_pred = gnb.predict(X_test)

    accuracy = accuracy_score(Y_test, Y_pred)
    conf_matrix = confusion_matrix(Y_test, Y_pred)

    print(f'Confusion matrix:\n', conf_matrix)
    print(f'Test accuracy:', accuracy)
    print(f'Test precision is', precision_score(Y_test, Y_pred, average='weighted'))
    print(f'Test recall is', recall_score(Y_test, Y_pred, average='weighted'))

    res = permutation_importance(gnb, X_train, Y_train, scoring='accuracy', n_repeats=20, random_state=42)
    p_importances = res['importances_mean'] / res['importances_mean'].sum()
    print(f"The permutation-based feature importance is {p_importances}")
