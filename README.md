# NeuralNetworks
Code and Notes

Some notes about NN:

Parts involved:
Activation Functions: many options
signmoid, hyperbolic tangent, logit
Back-propigation:
Uses gradient descent.  
SGD: 
Done in batches using matrix-matrix multiplications instead of updating weights after every case
Can adjust learning rate

Feed Forward:

Softmax:
imagine each node produces a logit log(z/(1-z)
provides probabilities to output that are constrained so sum to 1 (class probability)

What to watch out for:
overfitting (bias-variance tradoff).  
How to cope with overfitting:
* early stopping - if error worsening could revert to where trend started
* penalization - L1 or L2 penalty; will zero out nodes in network or constrain relationship w/ each other
* add noise to weights - will make model more robust

Performance improvement:
Optimizers: ADAM
RMSPROP
Regularization

Generally speaking want to reduce cross entropy error
