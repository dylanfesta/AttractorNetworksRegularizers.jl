# AttractorNetworksRegularizers.jl
Utility functions for optimization with regularizers

They solve the following problem:   several arrays of parameters, that follow different hard constraints need to be re-parametrized so that they are unconstrained and unbounded.  Also they are packed into a single vector that can be fed to an optimization function.

Gradients of parametrization are also computed, so that can be chained when computing the gradient.  
