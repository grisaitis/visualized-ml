# gaussian mixture modeling

## overview...

1. implement a class for gaussian mixtures
2. make a gaussian mixture; generate some data with it
3. use an algorithm to "learn" the original mixture from the data

## the actual machine learning...

In general, find parameters of the gaussian mixture that maximize a "goodness of fit" quantity.

1. Optimize with gradient descent
    - start with random parameters
    - compute their derivatives with respect to the fitness function
    - update the parameters bit by bit, in the direction of their gradients
    - repeat until things don't improve anymore
2. Optimize with "expectation-maximization"
    - start with random parameters
    - compute "fuzzy guesses" of which mixture component we think each data point belongs to
        - if we have N data points and K mixture components, then we compute N x K guesses
    - update mixture parameters given the "fuzzy guesses"
        - "parameters" here are each mixture component's mean, standard deviation, and "weight" in the mixture
        - "update" here maximizes the same fitness quantity as #1 (but we're not using gradient descent)
    - repeat until victory
