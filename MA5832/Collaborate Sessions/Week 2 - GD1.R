

#**************************************************************
#                          OBJECTIVES
#***************************************************************
# Goal is to write a Gradient descent function for univariate case,
# and implement the function to find the optimal solution of x for the 
# loss function x^2+2*x+5
#
# Key elememts: First derivatives of the loss function
# and learning rate (step.size).
#**************************************************************


#**************************************************************
#          INPUT OF A FUNCTION 
#**************************************************************
# f                  : an univariate loss function (e.g. f(x) = x^2+2*x+5)
# x0                 : starting points of x
# step_size          : also know as learning rate (often set at a small value, or 
#                      set at the inverse of Hessian matrix)
# max.iter           : the maximum number of iterations that you want to run the optimisation problem
# changes (optional) : if the gradient is smaller than the threshod ("changes"), stop
# go back and re-run the function before continuing the next iteration. 

install.packages('numDeriv')# This package helps to find the first 
#                             derivative of a function.
#                             In some cases, you will have to analytically derive
#                             the first and second derivatives by yourself.

library(numDeriv)

gradient.descent<-function(f, x0, step.size=0.05, max.iter=100, changes=0.001){

#Store the values of x across number of iterations
  x<-matrix(-99, ncol=length(x0), nrow=max.iter)
  
# Step 1 in Gradient Descent method  
  x[1,]<-x0
  
# Step 2 in Gradient Descent method  
  gradient<-grad(f, x0)
  
  for( i in 2:max.iter){
    
# Step 3 and 4 in Gradient Descent method  
    x[i,]<-x[i-1,]-step.size*gradient

# Updating the gradient     
    gradient<-grad(f, x[i,])
    
    if(i>1 & all(abs(gradient)< changes)){
      i=i-1
      break;
    }
  }
# Print out the results  
  return(list("x"=x[i,], x[1:i,]))
}


#***************************************************************
# An application 
#***************************************************************
loss<-function(x){
  return(x^2+2*x+5)
}
gradient.descent(loss, 1)
