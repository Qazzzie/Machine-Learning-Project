"""
    SVM Star classifier

"""

## Import packages  ##

from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
import csv 
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


## Import and sanitize data ##

color_dict = {           #Categorize the colors. Since it's a spectrum, one-hot encoding is not necessary.
    "blue": 0, # blue == 0
    'blue white': 1, 'yellowish white': 1, 'white': 1, 'whitish': 1,  # white == 1
    'pale yellow orange': 2, 'orange': 2, 'orange red': 2, 'yellow white': 2, "yellowish": 2, "white yellow" : 2, # orange or yellow == 2
    "red": 3, # red == 3
    }

spectral_dict = {'O':0, 'B':1, 'A':2, 'F':3, 'G':4, 'K':5, 'M':6}  #These correspond to a heat spectrum, so one-hot encoding is not necessary.

def sanitize_data(row):
    if all([row[0], row[1], row[2], row[3], row[4], row[5]]):    #removes any rows for which data is missing
        return [float(row[0]),                 #Average Temperature [K]
                float(row[1]),                 #Relative Luminosity
                float(row[2]),                 #Relative Radius
                float(row[3]),                 #Absolute Magnitude
                color_dict.get((str(row[4]).replace('-',' ').lower())),   #Color: improve format by removing hyphens and making lower-case
                spectral_dict.get((str(row[5]))) #Spectral Class
                ]
    else:
        return False    #if any data is missing, the data is not used

star_features = []
star_classes = []
 ##INSERT YOUR DATA FILE PATH ON THE NEXT LINE
data_file_path = 'C:\\Users\\morga\\GitHub\\Machine-Learning-Project\\stardata\\Stars.csv'
import csv
with open(data_file_path) as csvfile:
    reader = csv.reader(csvfile, delimiter =',')
    next(reader)
    for row in reader:
        data = sanitize_data(row)
        if data:
            star_features.append(data)       #add the input features to the features list
            star_classes.append(int(row[6])) #add the class to the class list

#Run statistics to find whether the dataset is balanced
X = np.array(star_features)
Y = np.array(star_classes)

# Split into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

m = len(Y_train)
n = len(X_train[0]) +1
print(f'Number of training instances is',m)
print(f'Number of features is',n)


## Data visualization playground
# plt.plot(X_train[:, 0][Y_train==0], X_train[:, 3][Y_train==0], 'rs', label="Red Dwarf")
# plt.plot(X_train[:, 0][Y_train==1], X_train[:, 3][Y_train==1], 'yo', label='Brown Dwarf')
# plt.plot(X_train[:, 0][Y_train==2], X_train[:, 3][Y_train==2], 'bs', label="White Dwarf")
# plt.plot(X_train[:, 0][Y_train==3], X_train[:, 3][Y_train==3], 'ro', label='Main Sequence')
# plt.plot(X_train[:, 0][Y_train==4], X_train[:, 3][Y_train==4], 'bo', label="Super Giants")
# plt.plot(X_train[:, 0][Y_train==5], X_train[:, 3][Y_train==5], 'ys', label="Hyper Giants")
# plt.xlabel('Normalized Average Temperature')
# plt.ylabel('Normalized Absolute Magnitude')
# plt.legend()


## Build SVM ##
svc = SVC()
params_ = {   #These are the parameters to test for the support vector machine
    "kernel": ['linear', 'poly','sigmoid', 'rbf'],
    "C": [0.1, 1, 10, 100], #1, 10
    "degree": [1, 2, 3]
}

star_svc = GridSearchCV(svc, params_, verbose=1, scoring = 'accuracy')   #Perform a grid search to find the best parameters, based on the accuracy score
star_svc.fit(X_train, Y_train)

Y_pred = star_svc.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Best parameters from the grid search were {star_svc.best_params_}")
print(f"Accuracy of training set is {star_svc.best_score_}")
print(f'Best test accuracy is',accuracy)
conf_matrix = confusion_matrix(Y_test,Y_pred)
print(f'Confusion matrix:\n',conf_matrix)