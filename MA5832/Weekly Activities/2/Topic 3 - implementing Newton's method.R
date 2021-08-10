library(numDeriv)

newton_meth <- function(f, x0, step_size = 0.05, max_iter = 100, conv_thresh = 0.001) {
  x = matrix(-99, ncol = length(x0), nrow = max_iter)
  x[1,] <- x0
  gradient <- grad(f, x0)
  hess_m <- hessian(f, x0)
  
  for(i in 2:max_iter){
    x[i,] <- x[i-1,] - step_size*gradient*solve(hess_m)
    gradient <- grad(f, x[i,])
    hess_m <- hessian(f, x[i,])
    
    if(i > 1 & all(abs(gradient) < conv_thresh)){
      i = i - 1
      break;
    }
  }
  return(list("x" = x[i,], x[1:i,]))
}


loss_func <- function(x){
  return(x^2 + 2*x + 5)
}

nm_results <- newton_meth(loss_func, 1)

library(tidyr)
nm_results2 <- tibble(nm_results) %>% unnest(cols = c(nm_results))
nm_results2 <- nm_results2[2:101,]