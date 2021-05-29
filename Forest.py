from sklearn.ensemble import RandomForestClassifier
import csv

with open(data_file_path) as csvfile:
    reader = csv.reader(csvfile, delimiter =',')
    next(reader)
    for row in reader:
        data = sanitize_data(row)
        if data:
            star_features.append(data)       #add the input features to the features list
            star_classes.append(int(row[6])) #add the class to the class list

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
