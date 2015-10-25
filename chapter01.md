###Chapter 1 Using neural nets to recognize handwritten digits


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
