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
