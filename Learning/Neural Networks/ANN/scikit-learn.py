from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix

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


mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))


mlp.fit(X_train,y_train)

print("ANN trained...")

predictions = mlp.predict(X_test)

print("\n\n====== Confusion Matrix =======\n")

print(confusion_matrix(y_test,predictions))

print("\n\n=======Classification Report==========\n")


print(classification_report(y_test,predictions))