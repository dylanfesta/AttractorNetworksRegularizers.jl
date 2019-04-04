# AttractorNetworksRegularizers.jl
Utility functions for optimization with regularizers

In case one needs to optimize many elements contained in various arrays with different hard constraints, this package packs them up into a single vector that can be optimized without constraints by some optimization function.

Gradients of parametrization are also computed, so that can be chained when computing the gradient.  

  
