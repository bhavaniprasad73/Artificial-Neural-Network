# importing the libraries needed in this project

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
# Data Preparation Stage

# Loading the data into a dataframe
data = pd.read_csv('C:/Users/ebhavaniprasad/Desktop/Data Mining/breast-cancer-wisconsin-data/data.csv')
print('Dimension of the dataset : ', data.shape)
print(data.head())


# Removing the empty column from the dataset
del data['Unnamed: 32']

# Separating the feature variables and class variable(target variable)

X = data.iloc[:, 2:].values       # Feature variable
Y = data.iloc[:, 1].values        # Actual class label
print(type(Y))
print("\n Actual Class Labels : ", Y)


# Class Label encoding M & B to 1 & 0
encoder = LabelEncoder()
Y = encoder.fit_transform(Y)
print('After Encoding : ', Y)

# Splitting data into test and training sets and randomly selecting in order to bias
# (sometimes they are highly correlated data)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state= 0)


# Scaling our training data (feature scaling)
# Each feature in our dataset now will have a mean = 0 and standard deviation = 1

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# ---------------------------------------------------------------------------------------------------------------

# Model Building Stage
# Building the Neural Network

print(Y[:,None].shape[1])

class NeuralNetwork:
    def __init__(self, X, Y, X_test, Y_test, hidden_nodes=12, learninig_rate=0.1, epochs=5000):
        # data

        self.X = X
        self.Y = Y[:, None]
        self.X_test = X_test
        self.Y_test = Y_test

        # defining parameters

        np.random.seed(4)
        self.input_nodes = len(X[0])     # number of features in the training data
        self.hidden_nodes = hidden_nodes
        self.output_npdes = self.Y.shape[1]
        self.learning_rate = learninig_rate

        # initializing the weights for our network

        self.w1 = 2 * np.random.random((self.input_nodes, self.hidden_nodes)) - 1
        self.w2 = 2 * np.random.random((self.hidden_nodes, self.output_npdes)) - 1

        self.train(epochs)  # Since we have to train our model for many times we here pass epochs count
        self.test()
        # in between input and hidden layers
    # Defining the activation function as a sigmoid function
    def sigmoid(self, X):
        return (1 / (1 + np.exp(-X)))

    def sigmoid_prime(self, X):
        return X * (1 - X)



    def train(self, epochs):

        for e in range(epochs):
            # FORWARDPROPAGATION
            l1 = self.sigmoid(np.dot(self.X, self.w1))
            # in between hidden and output
            l2 = self.sigmoid(np.dot(l1, self.w2))

            # BACKPROPAGATION
            # Network error (True value - Predicted value)

            error = self.Y - l2

            # error for each of the layers

            l2_delta = error * self.sigmoid_prime(l2)
            l1_delta = l2_delta.dot(self.w2.T) * self.sigmoid_prime(l1)

            self.w2 = np.add(self.w2, l1.T.dot(l2_delta) * self.learning_rate)
            self.w1 = np.add(self.w1, self.X.T.dot(l1_delta) * self.learning_rate)

        print('Error : ', (abs(error)).mean())

    # testing and evaluation


    def test(self):
        correct = 0
        pred_list = []
        l1 = self.sigmoid(np.dot(self.X_test, self.w1))
        l2 = self.sigmoid(np.dot(l1, self.w2))

        for i in range(len(l2)):
            if l2[i] >= 0.5:
                pred = 1
            else:
                pred = 0

            if pred == self.Y_test[i]:
                correct += 1

            pred_list.append(pred)

        print('Test Accuracy : ', ((correct / len(Y_test)) * 100), '%')



        precision, recall,fscore, support =  precision_recall_fscore_support(Y_test, pred_list, average=None)

        tn, fp, fn, tp = confusion_matrix(Y_test, pred_list).ravel()
        print('True Nagative', tn)
        print('False Positive', fp)
        print('False Negative', fn)
        print('True Positive', tp)

        total = tn + tp + fn + fp

        print('Test Accuracy : ', (tn + tp)/total)
        print('Missclassification Rate : ', (fn + fp)/total)
        print('precision : ', precision)
        print('recall : ', recall)
        print('FScore : ', fscore)
        print('Support : ', support)
        print('Sensitivity or TPR : ', (tp/ (tp + fn)) )
        print('Specificity or TNR : ', (tn/(tn+fp)))
        print('False Positive Rate or Fallout : ', (fp/(fp+tn)))
        print('False Negative Rate : ', (fn/(fn+tp)))

        print('False Discovery Rate : ', (fp/(tp+fp)))

        cm = confusion_matrix(Y_test, pred_list)
        sns.heatmap(cm, annot=True)
        plt.savefig('h.png')
        plt.show()


nn = NeuralNetwork(X_train, Y_train, X_test, Y_test)
































