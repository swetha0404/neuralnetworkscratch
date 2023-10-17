from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation):
        np.random.seed(1)

        self.activation = activation

        # initialize weights for layers with random values and initialize bais terms for input and hidden layers
        # l0 is input layer, l1 is hidden layer, and l2 is output layer
        # all weights from input layer to hidden layer
        self.l0_l1_weights = 2 * np.random.random((input_size, hidden_size)) - 1
        # hidden later bias connection
        self.l1_bias = np.zeros((1, hidden_size))
        # all weights from hidden layer to output layer
        self.l1_l2_weights = 2 * np.random.random((hidden_size, output_size)) - 1
        # output layer bias connection
        self.l2_bias = np.zeros((output_size, 1))

        # initialize weights and bias for adagrad optimization calculation
        self.l0_l1_weights_adagrad = np.zeros_like(self.l0_l1_weights)
        self.l1_bias_adagrad = np.zeros_like(self.l1_bias)
        self.l1_l2_weights_adagrad = np.zeros_like(self.l1_l2_weights)
        self.l2_bias_adagrad = np.zeros_like(self.l2_bias)

    # sigmoid function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # sigmoid derivative function
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # tanh function
    def tanh(self, x):
        return ((np.exp(x)-np.exp(-x)) / (np.exp(x) + np.exp(-x)))

    # tanh derivative function
    def tanh_derivative(self, x):
        return 1 - x ** 2

    # relu function
    def relu(self, x):
        return np.maximum(0, x)

    #relu derivative function
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def preprocess(self, X):
        # drop null values and duplicates
        X = X.dropna()
        X = X.drop_duplicates()

        #  one hot encoding for catergorical attribute 'Sex'
        X = pd.get_dummies(X, columns=['Sex'])
        X['Sex_F'].replace([False, True],[0, 1], inplace=True)
        X['Sex_I'].replace([False, True],[0, 1], inplace=True)
        X['Sex_M'].replace([False, True],[0, 1], inplace=True)

        # normalize and scale X
        X = (X - X.mean()) / X.std()
        X = np.c_[np.ones(X.shape[0]), X]

        return X

    # forward pass
    def forward_pass(self, X):
        # find output for hidden layer using inputs and weights and bias for hidden layer
        # l1_input is the input to the hidden layer
        self.l1_input = np.dot(X, self.l0_l1_weights) + self.l1_bias
        if self.activation == "sigmoid":
            self.l1_output = self.sigmoid(self.l1_input)
        elif self.activation == "tanh":
            self.l1_output = self.tanh(self.l1_input)
        elif self.activation == "relu":
            self.l1_output = self.relu(self.l1_input)

        # find output for output layer using output from hidden later and weights and bias for output layer
        # l2_input is the input to the output layer
        self.l2_input = np.dot(self.l1_output, self.l1_l2_weights) + self.l2_bias
        if self.activation == "sigmoid":
            output = self.sigmoid(self.l2_input)
        elif self.activation == "tanh":
            output = self.tanh(self.l2_input)
        elif self.activation == "relu":
            output = self.relu(self.l2_input)

        return output

    def backward_pass(self, X, y, output, learning_rate):
        # do backward pass for output layer
        error_l2 = y - output
        # delta_l2 is the delta for output layer (delta = error * derivative of activation function)
        if self.activation == "sigmoid":
          delta_l2 = error_l2 * self.sigmoid_derivative(output)
        elif self.activation == "tanh":
          delta_l2 = error_l2 * self.tanh_derivative(output)
        elif self.activation == "relu":
          delta_l2 = error_l2 * self.relu_derivative(output)
        # make output into a matrix from a numpy array
        delta_l2 = delta_l2.to_numpy().reshape(len(delta_l2), 1)

        # do backward pass for hidden layer
        error_l1 = delta_l2.dot(self.l1_l2_weights.T)
        # delta_l1 is the delta for hidden layer (delta = error * derivative of activation function)
        if self.activation == "sigmoid":
            delta_l1 = error_l1 * self.sigmoid_derivative(self.l1_output)
        elif self.activation == "tanh":
            delta_l1 = error_l1 * self.tanh_derivative(self.l1_output)
        elif self.activation == "relu":
            delta_l1 = error_l1 * self.relu_derivative(self.l1_output)

        # calculates the Adagrad updates for output and hidden layer weights and biases
        self.l1_l2_weights_adagrad += np.square(self.l1_output.T.dot(delta_l2))
        self.l2_bias_adagrad += np.square(np.sum(delta_l2, axis=0, keepdims=True))
        self.l0_l1_weights_adagrad += np.square(X.T.dot(delta_l1))
        self.l1_bias_adagrad += np.square(np.sum(delta_l1, axis=0, keepdims=True))

        # update weights and biases for the output and hidden layers using Adagrad
        self.l1_l2_weights+= (self.l1_output.T.dot(delta_l2) / (np.sqrt(self.l1_l2_weights_adagrad) + 1e-8)) * learning_rate
        self.l2_bias += (np.sum(delta_l2, axis=0, keepdims=True) / (np.sqrt(self.l2_bias_adagrad) + 1e-8)) * learning_rate
        self.l0_l1_weights += (X.T.dot(delta_l1) / (np.sqrt(self.l0_l1_weights_adagrad) + 1e-8)) * learning_rate
        self.l1_bias += (np.sum(delta_l1, axis=0, keepdims=True) / (np.sqrt(self.l1_bias_adagrad) + 1e-8)) * learning_rate

    def train(self, X, y, iterations, learning_rate):
        # preprocess X
        X = self.preprocess(X)
        # compute forward and backward pass for each iteration
        for iteration in range(iterations):
            output = self.forward_pass(X)
            output = output.flatten()
            self.backward_pass(X, y, output, learning_rate)
        y = (y - y.mean()) / y.std()
        # print training error
        print(f"Train Error MSE: %.5f" %mse(y, output))

    def test(self, X):
        # preprocess X and return output from one forward pass
        X = self.preprocess(X)
        return self.forward_pass(X)
    
def mse(y1, y2):
    return np.mean((y1 - y2)**2)


#Main function to read data, perform dataspecific preprocessing, calling Neural Network with specific hyperparameters
def main():

    url = 'https://github.com/st9488/CS6375_Assignment1/blob/fb56e27f3e3a43fefa56167d4697d94556b703dd/abalone.data?raw=true'
    df = pd.read_csv(url)

    # rename the columns
    df.rename(columns = {'Rings':'Age', 'Whole weight':'Whole', 'Shucked weight': 'Shucked', 'Viscera weight' : 'Viscera',  'Shell weight' : 'Shell'}, inplace = True)

    # setting y as target variable matrix
    y = df['Age']

    # dropping target variable from dataframe
    X = df.drop(['Age'], axis=1)

    # split the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1, shuffle=True)

    # create a neural network with input neurons, 4 hidden neurons, and output neurons an activation function
    input_size = len(X_train.to_numpy()[0])
    # check if output has multiple parameters
    if (type(y_train[0]) != np.ndarray):
        output_size = 1
    else:
        output_size = len(y_train[0])
    hidden_size = 4
    neural_net = NeuralNetwork(input_size+3, hidden_size, output_size, activation="sigmoid")

    # train the neural network for given iterations and learning rate
    neural_net.train(X_train, y_train, iterations=100, learning_rate=0.0001)

    # test and predict
    y_pred = neural_net.test(X_test)
    y_test = (y_test - y_test.mean()) / y_test.std()
    print("Test Error MSE: %.5f" %mse(y_test, y_pred.flatten()))

main() #Calling main function.