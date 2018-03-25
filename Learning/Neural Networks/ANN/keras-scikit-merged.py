"""
Example mentioned in scikit-learn-code.py of Breast cancer classification done using Keras.
"""

import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
import sys
cancer = load_breast_cancer()

print("loaded the data...")

X = cancer['data']
y = cancer['target']

X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)




# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Importing the Keras libraries and packages

#Initializing Neural Network
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 70, init = 'uniform', activation = 'relu', input_dim = 30))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 70, init = 'uniform', activation = 'relu'))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 40, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling Neural Network
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy')

print("NN init done..")

# Fitting our model 
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 500)

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)

print("\n\n====== Confusion Matrix =======\n")

print(confusion_matrix(y_test,y_pred))

print("\n\n=======Classification Report==========\n")


print(classification_report(y_test,y_pred))