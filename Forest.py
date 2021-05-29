from sklearn.ensemble import RandomForestClassifier
import utilities
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score

X_train, X_test, Y_train, Y_test = utilities.get_data_as_train_test_split(test_size=.33, random_state=None)
Random_Forest = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
Random_Forest.fit(X_train, Y_train)
Y_pred = Random_Forest.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
confusion_matrix = confusion_matrix(Y_test, Y_pred)

print(f'Test accuracy:', accuracy)
print(f'Test precision is', precision_score(Y_test, Y_pred, average='weighted'))
print(f'Test recall is', recall_score(Y_test, Y_pred, average='weighted'))
print(f'Confusion matrix:\n', confusion_matrix)
