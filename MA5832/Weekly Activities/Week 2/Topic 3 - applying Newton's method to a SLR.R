library(datarium)

data <- marketing[1:20,c(1,4)]
plot(data$youtube, data$sales, col = "blue", pch = 20)

# lm function results
lm_mod <- lm(sales ~ youtube, data = data)
lm_mod$coefficients

# Newton's method algorithm
newton_meth <- function(x, y, p0, max_iter = 150000, conv_thresh = 0.01) {
  p <- matrix(0, nrow = max_iter, ncol = length(p0))
  gradient <- matrix(0, nrow = max_iter, ncol = length(p0))
  hessian <- matrix(0, nrow = length(p0), ncol = length(p0))
  p[1,] <- p0
  for(i in 1:(max_iter-1)) {
    yhat <- p[i,1]*x + p[i,2]
    gradient[i,1] <- -2*mean(x*(y - yhat))
    gradient[i,2] <- -2*mean(y - yhat)
    hessian[1,1] <- 2*mean(x^2)
    hessian[2,1] <- 0
    hessian[1,2] <- 0
    hessian[2,2] <- 2
    
    p[i+1,] <- p[i,] - t(solve(hessian) %*% gradient[i,])
    }
  return(list("i" = i, "final" = p[i,], "p" = p))
}

# implement Newton's method algorithm
x <- data$youtube
y <- data$sales
p0 <- c(0.04, 8.0)

# changing value of max_iter but keeping default for conv_thresh
nm_results <- newton_meth(x, y, p0, max_iter = 100000)
