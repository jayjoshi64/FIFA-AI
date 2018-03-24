### Simple ANN in Scikit learn

We'll use SciKit Learn's built in Breast Cancer Data Set which has several features of tumors with a labeled class indicating whether the tumor was Malignant or Benign. We will try to create a neural network model that can take in these features and attempt to predict malignant or benign labels for tumors it has not seen before. Let's go ahead and start by getting the data!

    >>> from sklearn.datasets import load_breast_cancer
    >>> cancer = load_breast_cancer()

This object is like a dictionary, it contains a description of the data and the features and targets:

    >>> cancer.keys()

    dict_keys(['DESCR', 'feature_names', 'target_names', 'target', 'data'])

<p>

    # Print full description by running:
    # print(cancer['DESCR'])
    # 569 data points with 30 features
    >>> cancer['data'].shape

    (569, 30)

Let's set up our Data and our Labels:

    X = cancer['data']
    y = cancer['target']

##### Train Test Split
 
Let's split our data into training and testing sets, this is done easily with SciKit Learn's train_test_split function from model_selection:

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y)


##### Data Preprocessing

 
The neural network may have difficulty converging before the maximum number of iterations allowed if the data is not normalized. Multi-layer Perceptron is sensitive to feature scaling, so it is highly recommended to scale your data. Note that you must apply the same scaling to the test set for meaningful results. There are a lot of different methods for normalization of data, we will use the built-in StandardScaler for standardization.

    >>> from sklearn.preprocessing import StandardScaler
    >>> scaler = StandardScaler()
    >>> # Fit only to the training data
    >>> scaler.fit(X_train)


    StandardScaler(copy=True, with_mean=True, with_std=True)

<p>

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


##### Training the model

 
Now it is time to train our model. SciKit Learn makes this incredibly easy, by using estimator objects. In this case we will import our estimator (the Multi-Layer Perceptron Classifier model) from the neural_network library of SciKit-Learn!

    >>> from sklearn.neural_network import MLPClassifier

Next we create an instance of the model, there are a lot of parameters you can choose to define and customize here, we will only define the hidden_layer_sizes. For this parameter you pass in a tuple consisting of the number of neurons you want at each layer, where the nth entry in the tuple represents the number of neurons in the nth layer of the MLP model. There are many ways to choose these numbers, but for simplicity we will choose 3 layers with the same number of neurons as there are features in our data set:

    >>> mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))


Now that the model has been made we can fit the training data to our model, remember that this data has already been processed and scaled:

    >>> mlp.fit(X_train,y_train)

    MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(30, 30, 30), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)


You can see the output that shows the default values of the other parameters in the model. I encourage you to play around with them and discover what effects they have on your model!

##### Predictions and Evaluation

 
Now that we have a model it is time to use it to get predictions! We can do this simply with the predict() method off of our fitted model:

    >>> predictions = mlp.predict(X_test)

---



Now we can use SciKit-Learn's built in metrics such as a classification report and confusion matrix to evaluate how well our model performed:

    >>> from sklearn.metrics import classification_report,confusion_matrix
    >>> print(confusion_matrix(y_test,predictions))

    [
        [50  3]
        [ 0 90]
    ]

<p>

    >>> print(classification_report(y_test,predictions))

        precision    recall  f1-score   support

          0       1.00      0.94      0.97        53
          1       0.97      1.00      0.98        90

        avg / total       0.98      0.98      0.98       143


---

If you  want to extract the MLP weights and biases after training your model, you use its public attributes coefs_ and intercepts_.

coefs_ is a list of weight matrices, where weight matrix at index i represents the weights between layer i and layer i+1.

intercepts_ is a list of bias vectors, where the vector at index i represents the bias values added to layer i+1.

    >>>len(mlp.coefs_)

    4

<p>

    >>> len(mlp.coefs_[0])

    30

<p>

    >>> len(mlp.intercepts_[0])

    30