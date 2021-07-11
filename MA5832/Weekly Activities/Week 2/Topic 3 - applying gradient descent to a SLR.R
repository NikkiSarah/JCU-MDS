library(datarium)

data <- marketing[1:20,c(1,4)]
plot(data$youtube, data$sales, col = "blue", pch = 20)

# lm function results
lm_mod <- lm(sales ~ youtube, data = data)
lm_mod$coefficients

# gradient descent algorithm
grad_desc <- function(x, y, p0, step_size, max_iter, conv_thresh) {
  p <- matrix(0, nrow = max_iter, ncol = length(p0))
  gradient <- matrix(0, nrow = max_iter, ncol = length(p0))
  p[1,] <- p0
  for(i in 1:(max_iter-1)) {
    yhat <- p[i,1]*x + p[i,2]
    gradient[i,1] <- -2*mean(x*(y - yhat))
    gradient[i,2] <- -2*mean(y - yhat)
    p[i+1,1] <- p[i,1] - step_size*gradient[i,1]
    p[i+1,2] <- p[i,2] - step_size*gradient[i,2]
    if(i > 1 & all(abs(gradient[i,]) < conv_thresh)) {
      i = i - 1
      break;
    }
  }
  return(list("i" = i, "p" = p, "g" = gradient))
}

# implement gradient descent algorithm
x <- data$youtube
y <- data$sales
p0 <- c(0.1, 8.0)

# setting values for step_size, max_iter, conv_thresh
gd_results <- grad_desc(x, y, p0, step_size = 0.000003, max_iter = 150000, conv_thresh = 0.01)
gd_results$p[gd_results$i,]