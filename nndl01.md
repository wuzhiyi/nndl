##Chapter 1 Using neural nets to recognize handwritten digits

Two important types of artificial neuron:

- Perceptron
- Sigmoid neuron

Standard learning algorithm for neural networks:

- Stochastic gradient descent

###Perceptrons
We proposed a simple rule to compute the output, which is _weights_, _w1_, _w2_, real numbers expressing the importance of the respective inputs to the output. The neuron's output, 0 or 1, is determined by whether the weighted sum ∑wjxj is less than or greater than some _threshold value_. Just like the weights, the threshold is a real number which is a parameter of the neuron.</br>
To put it in more precise algebraic terms:

    if sum(w[j]x[j]) <= threshold, output = 0;
    if sum(w[j]x[j]) >  threshold, output = 1;
    
Simplify above by using two notational changes:</br>

1. use _w⋅x_ ≡ ∑w[j]x[j], where _w_ and _x_ are vectors whose components are the weights and inputs, respectively
2. move the threshold to the order side of the inequality, and to replace it by what's known as the perceptron's _bias_, b ≡ -threshold.

The perceptron rule can be rewritten:

    output = 0, if _w⋅x_ + b <= 0;
    output = 1, if _w⋅x_ + b >  0;
    
###Sigmoid neurons
Sigmoid neurons are similar to perceptrons, but modified so that small changes in their weights and bias cause only a small change in their output. That's the crucial fact which will allow a network of sigmoid neurons to learn.
Just like a perceptron, the sigmoid neuron has weights for each input, _w1_, _w2_, ..., and an overall bias, _b_. But the output is not 0 or 1. Instead, it's σ(_w⋅x_ + _b_), where σ is called the _sigmoid function_.
The output of a sigmoid neuron with inputs x1, x2, ..., weights w1, w2, ..., and bias _b_ is

    b = 1/(1 + exp(-∑w[j]x[j] -b));
    
The smoothness of σ means that small changes Δ_w[j]_ in the weights and Δ_b_ in the bias will produce a small change Δoutput in the output from the neuron. In fact, calculs tells us that Δoutput is well approximated by

    Δoutput ≈ ∑(∂output/∂w[j])Δw[j] + ∑(∂output/∂b)Δb;

where the sum is over the weights, _w[j]_, and ∂output/∂_w[j]_ and ∂output/∂_b_ denote partial derivatives of the output with respect to _w[j]_ and _b_ respectively.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      

###The architecture of neural networks

The leftmost layer in network is called the __input layer__, and the neurons within the layer are called _input neurons_.</br>
The rightmost or _output_ layer contains the _output neurons_.
The middle layer is called a _hidden layers_, since the neurons in this layer are neither inputs nor outputs.
(Multiple layer networks are sometimes called _multipayer perceptrons_ or _MLPs_.)</br>
The neural networks where the output from one layer is used as input to the next layer. Such networks are called _feedback_ neural networks. This means there are no loops in the network - information is always fed forward, never fed back.
There are other models of artificial neural networks in which feedback loops are possible. These models are called _recurrent neural networks_. The idea in these models is to have neurons which fire for some limited duration of time, before becoming quiescent.

###Learning with gradient descent
What we'd like is an algorithm which lets us find weights and biases so that the output from the network approximates y(x) for all training inputs _x_. To quanitfy how well we're achieving this goal we define a _cost function_:

    C(w,b) ≡ (∑||y(x)-a||^2))/2n;
    
Here, _w_ denotes the collection of all weights in the network, _b_ all the biases, _n_ is the total number of training inputs, _a_ is the vector of outputs from the network when _x_ is input, and the sum is over all training inputs, _x_. The notation ||_v_|| just denotes the usual length function for vector _v_. We'll call C the quadratic cost function; it's also sometimes known as the mean squared error or just MSE. Inspecting the form of the quadratic cost function, we see that C(_w_,_b_) is non-negative, since every term in the sum is non-negative.
Suppose in particular that _C_ is a function of _m_ variables, _v1_, ..., _vm_. Then the change Δ_C_ in _C_ produced by a small change Δ_v_=(Δ_v_,...,Δ_vm_)^T is 

    ΔC ≈ ∇C⋅Δv;

where the gradient ∇_C_ is the vector

    ∇C ≡ (∂C/∂v1, ..., ∂C/∂vm)^T;

Just as for the two variable case, we can choose

    Δv = −η∇C;

and we're guaranteed that our (approximate) expression for Δ_C_ will be negative. This gives us a way of following the gradient to a minimum, even when _C_ is a function of many variables, by repeatedly applying the update rule

    v → v' = v - η∇C;
    
_Stochastic gradient descent_ can ben used to speed up learning. The idea is to estimate the gradient ∇_C_ by computing ∇_Cx_ for a small sample of randomly chosen training inputs.
To make these ideas more precise, stochastic descent works by randomly picking out a small number _m_ of randomly chosen training inputs. We'll label those random training inputs X1, X2, ..., Xm, and refer to them as a mini-batch. Provided the sample size _m_ is large enough we expect that the average value of the ∇C[X[j]] will be roughly equal to the average over all ∇C[x], that is,

    (∑∇C[X[j]])/m ≈  (∑∇C[x])/n = ∇C

where the second sum is over the entire set of training data.
To connect this explicitly to learning in neural networks, suppose _wk_ and _bl_ denote the weights and biases in our neural network. Then stochasitc gradient descent works by picking out a randomly chosen mini-batch of training inputs, and training with those,

    wk → wk' = wk - (η/m)*∑(∂C[X[j]]/∂w[k])
    bl → bl' = bl - (η/m)*∑(∂C[X[j]]/∂b[l])
    
###Implementing our network to classify digits
The centerpiece is a `Network` class, which we use to represent a neural network. Here's the code to initialize a `Network` object:

    class Network(object):
    	def __init__(self,sizes):
    		self.num_layers = len(sizes)
    		self.size    = sizes
    		self.biases  = [np.random.randn(y,1) for y in sizes[1:]]
    		self.weights = [np.random.randn(y,x) 
    						for x,y in zip(sizes[:-1],sizes[1:])]

In this code, the list `sizes` contains the number of neurons in the respective layer. For example, if we want to create a `Network` object with 2 neurons in the first layer, 3 neurons in the second layer, and 1 neuron in the final layer, do as following:

    net = Network([2,3,1])

Define the sigmoid function:

    def sigmoid(z):
    	return 1.0/(1.0+np.exp(-z))

Add a `feedforward` method to the `Network` class, which, given an input `a` for the network, returns the corresponding output:
    def feedforward(self, a):
	"""Return the output of the network if "a" is input."""
	for b,w in zip(self.biases, self.weights):
		a = sigmoid(np.dot(w,a)+b)
	return a

Give `Network` objects an `SGD` method which implements stochastic gradient descent, here's the code:

	def SGD(self, training_data, epochs, mini_batch_size, eta, 
        test_data_=None):
    """Train the neural network using mini-batch stochastic
    gradient descent. The "training_data" is a list of tuples
    "(x,y)" representing the training inputs and the desired
    outputs. The other non-optional parameters are self-
    explanatory. If "test_data" is provided then the network
    will be evaluated against the test data after each epoch,
    and partial progress printed out. This is useful for 
    tracking progress, but slows things down substantially."""
    if test_data: n_test = len(test_data)
    n = len(training_data)
    for j in xrange(epochs):
        random.shuffle(training_data)
        mini_batches = [
            training_data[k:k+mini_batch_size]
            for k in xrange(0, n, mini_batch_size)]
        for mini_batch in mini_batches:
            self.update_mini_batch(mini_batch, eta)
        if test_data:
            print "Epoch {0}: {1}/{2}".format(
                j, self.evaluate(test_data), n_test)
        else:
            print "Epoch {0} complete".format(j)
            
Here's the code for the `update_mini_batch`method:

    def update_mini_batch(self, mini_batch, eta):
    """Update the network's weights and biases by apllying
    gradient descent using backpropagation to a single mini
    batch. The "mini_batch" is a list of tuples "(x,y)",
    and "eta" is the learning rate."""
    nabla_b = [np.zeros(b.shape) for b in self.baises]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    for x,y in mini_batch:
        delta_nabla_b, delta_nala_w = self.backprop(x,y)
        nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
    self.weights = [w-(eta/len(mini_batch))*nw
                    for w, nw in zip(self.weights, nabla_w)]
    self.biases = [b-(eta/len(mini_batch))*nb
                    for b, nb in zip(self.biases, nabla_b)]

Load in the MNIST data by executing the following commands in Python shell:

	>>> import mnist_loader
	>>> training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

Set up a `Network` with 30 hidden neurons:

	>>> import network
	>>> net = network.Network([784, 30, 10])

Use stochastic gradient descent to learn from the MNIST `training_data` over 30 epochs, with a mini-batch size of 10, and learning rate of η=3.0:

	>>> net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
	
##Chapter 2 How the backpropagation algorithm works

###Wram up: a fast matrix-based approach to computing the output from a neural network
