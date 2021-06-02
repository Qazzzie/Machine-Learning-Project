## Data visualization playground

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
data_file_path = 'C:\\Users\morga\\GitHub\\Machine-Learning-Project\\stardata\\Stars.csv' #stardata\\Stars.csv
import csv
with open(data_file_path) as csvfile:
    reader = csv.reader(csvfile, delimiter =',')
    next(reader)
    for row in reader:
        data = sanitize_data(row)
        if data:
            star_features.append(data)       #add the input features to the features list
            star_classes.append(int(row[6])) #add the class to the class list

X = np.array(star_features)
Y = np.array(star_classes)

# Split into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

plt.plot(X_train[:, 4][Y_train==0], X_train[:, 3][Y_train==0], 'ro', label="Red Dwarf")
plt.plot(X_train[:, 4][Y_train==1], X_train[:, 3][Y_train==1], 'mo', label='Brown Dwarf')
plt.plot(X_train[:, 4][Y_train==2], X_train[:, 3][Y_train==2], 'co', label="White Dwarf")
plt.plot(X_train[:, 4][Y_train==3], X_train[:, 3][Y_train==3], 'go', label='Main Sequence')
plt.plot(X_train[:, 4][Y_train==4], X_train[:, 3][Y_train==4], 'ko', label="Super Giants")
plt.plot(X_train[:, 4][Y_train==5], X_train[:, 3][Y_train==5], 'yo', label="Hyper Giants")
plt.xlabel('Color Spectrum')
plt.ylabel('Normalized Absolute Magnitude')
plt.legend(loc=5,fontsize='small')