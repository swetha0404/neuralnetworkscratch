# neuralnetworkscratch
<h3> Python code for neural network from scratch for Abalone dataset. </h3>

This Neural Network's preprocessor function is tailored for the Abalone Dataset from UCI Machine Learning Repository. It can be edited accordingly for different datasets.

This is a starter code in Python 3.6 for a 2-hidden-layer neural network with three different options of activation functions.   
- Sigmoid
- Tanh
- Relu

The neural network is coded from scratch, without using any framework or ML libraries. But data processing uses pandas, splitting dataset uses sklearn's train_test_split, and the matrix operations are performed using numpy.

You need to have numpy and pandas installed before running this code.   

Then number of hidden layers are fixed (2), but the number of neurons in the hidden layers, number of iterations, learning rate, and activation functions could be changed in the main function.

Below are the meaning of symbols:

1. **train** : training dataset - can be a link to a URL or a local file <br>
   $~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$ - you can assume the last column will the label column <br>
    
2. **test** : test dataset   - can be a link to a URL or a local file <br>
   $~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$ - you can assume the last column will the label column <br>
3. **h1** : number of neurons in the first hidden layer <br>
4. **h2** : number of neurons in the second hidden layer <br>
5. **X** : vector of features for each instance <br>
6. **y** : output for each instance <br>
7. **w01, delta01, X01** : weights, updates and outputs for connection from layer 0 (input) to layer 1 (first hidden) <br>
8. **w12, delata12, X12** : weights, updates and outputs for connection from layer 1 (first hidden) to layer 2 (second hidden) <br>
9. **w23, delta23, X23** : weights, updates and outputs for connection from layer 2 (second hidden) to layer 3 (output layer) <br>
