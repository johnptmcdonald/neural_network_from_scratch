import numpy as np

np.random.seed(123)

def sigmoid(x):
	return 1.0/(1+np.exp(-x))

def sigmoid_derivative(x):
	return x * (1.0 - x)

# def relu(x):
#     return x * (x > 0)

# def relu_derivative(x):
#     return 1. * (x > 0)

class NeuralNetwork:
	def __init__(self, x, y):
		self.input = x
		print(f'Input:\n {x}\n')
		self.weights1 = np.random.rand(self.input.shape[1],2)
		print(f'''first layer weights
Each column represents weights INTO layer1 neuron
This is the same as each row representing weights out of an input layer neuron:\n {self.weights1}\n''')
		self.weights2 = np.random.rand(2,1)
		print(f'second layer weights:\n {self.weights2}\n')
		self.y = y
		self.output = np.zeros(self.y.shape)


	def feedforward(self, test=None):
		if test: self.input = test
		self.layer1 = sigmoid(np.dot(self.input, self.weights1))
		print(f'self.layer1 ():\n {self.layer1}')
		self.output = sigmoid(np.dot(self.layer1, self.weights2))
		return self.output

	def backprop(self):
		d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))

		d_weights1 = np.dot(self.input.T, (np.dot(2*(self.y - self.output) 
			* sigmoid_derivative(self.output), self.weights2.T) 
			* sigmoid_derivative(self.layer1)))

		self.weights1 += d_weights1
		self.weights2 += d_weights2

if __name__ == '__main__':
	X = np.array([[9,1,1],
				  [8,8,1],
				  [9,9,1],
				  [2,1,1]])

	print(X.shape)


	y = np.array([[0],
				  [1],
				  [1],
				  [0]])

	nn = NeuralNetwork(X,y)

	for i in range(1500):
		nn.feedforward()
		nn.backprop()

	print(nn.weights1)
	print(nn.feedforward())

"""
For a neural network of 3 input nodes, a single hidden layer of 2 nodes, and a single output layer:

Input X represents four inputs (each consisting of three integers)
Output y is the actual label for the four inputs
 
  INPUT 	OUTPUT
[[9,1,1],	 [[0],
 [8,8,1],	  [1],
 [9,9,1],	  [1],
 [2,1,1]]	  [0],

weights1 represents the weights from the input layer to the hidden layer

		WEIGHTS1 
[[0.69646919, 0.28613933] 	--- each row represents the weights of the connections coming out of an input neuron
 [0.22685145, 0.55131477]
 [0.71946897, 0.42310646]]

	|
	|
  Each column represents the weights of the connections entering a hidden layer node


As an example, let's take the first input [9,1,1] and run it through the network (feedforward) with the randomly initialized weights





"""





