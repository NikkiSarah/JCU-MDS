library(numDeriv)

# initialise gradient descent function with variables for the function to be solved, x0,
# step-size or learning rate, the maximum number of iterations and the convergence threshold.
grad_desc <- function(f, x0, step_size = 0.05, max_iter = 100, conv_thresh = 0.001) {
  # initialise the matrix of values
  x <- matrix(-99, ncol = length(x0), nrow = max_iter)
  # set the first value in the matrix as x0
  x[1,] <- x0
  # define the gradient/slope as a function of f and x0
  gradient <- grad(f, x0)
  
  # start looping over each value of x (except for x0)
  for(i in 2:max_iter){
    x[i,] <- x[i-1,] - step_size*gradient
    gradient <- grad(f, x[i,])
    
    # loop stops when the difference between the gradient values is less than the specified
    # threshold
    if(i > 1 & all(abs(gradient) < conv_thresh)){
      i = i - 1
      break;
    }
  }
  # print the values of x for the function
  return(list("x" = x[i,], x[1:i,]))
}

# define the loss function i.e the function being solved
loss_func <- function(x){
  return(x^2 + 2*x + 5)
}

# run the algorithm
gd_results <- grad_desc(loss_func, 1)

library(tidyr)
gd_results2 <- tibble(gd_results) %>% unnest(cols = c(gd_results))
gd_results2 <- gd_results2[2:80,]