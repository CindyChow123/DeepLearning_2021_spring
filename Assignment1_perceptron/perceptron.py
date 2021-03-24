import numpy as np
from sklearn.utils import shuffle

class Perceptron(object):

    def __init__(self, n_inputs, max_epochs=1e2, learning_rate=1e-2):
        """
        Initializes perceptron object.
        Args:
            n_inputs: number of inputs.
            max_epochs: maximum number of training cycles.
            learning_rate: magnitude of weight changes at each training cycle
        """
        self.n_inputs = n_inputs
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.w=np.zeros(3)
    def forward(self, input):
        """
        Predict label from input 
        Args:
            input: array of dimension equal to n_inputs.
        """
        label = np.ones(self.n_inputs)
        for i in range(len(input)):
            x=np.concatenate((input[i],[1]),axis=0)
            pro = np.dot(self.w,x)
            if pro < 0:
                label[i] = -1
        return label
        
    def train(self, training_inputs, labels):
        """
        Train the perceptron
        Args:
            training_inputs: list of numpy arrays of training points.
            labels: arrays of expected output value for the corresponding point in training_inputs.
        """
        for i in range(int(self.max_epochs)):
            x_train,y_train = shuffle(training_inputs,labels,random_state=10)
            for j in range(len(x_train)):
                x=np.concatenate((x_train[j],[1]),axis=0)
                loss = y_train[j] * np.dot(self.w,x)
                if loss <= 0:
                    self.w = self.w + self.learning_rate * y_train[j] * x

    def test(self, test_inputs, labels):
        pred = self.forward(test_inputs)
        correct = 0
        for i in range(len(labels)):
            if pred[i] == labels[i]:
                correct += 1
        perc = correct/len((labels))
        return perc, pred

