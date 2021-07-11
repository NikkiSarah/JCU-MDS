gd.lm<-function(x_1, x_2, y, p0, step.size= 0.00002, max.iter=150000, changes=0.001){
  
  p<-matrix(0, nrow=max.iter, ncol=length(p0))
  loss<-rep(NA, max.iter)
  gradient<-matrix(0, nrow=max.iter, ncol=length(p0))
  p[1,]<-p0
  for( i in 1:(max.iter-1)){
    yhat<-p[i,1]*x_1+p[i,2]*x_2
    gradient[i,1]<--2*mean(x_1*(y-yhat))
    gradient[i,2]<--2*mean(x_2*(y-yhat))
    p[i+1,1]<-p[i,1]-step.size*gradient[i,1]
    p[i+1,2]<-p[i,2]-step.size*gradient[i,2]
    loss[i]<-mean((yhat-y)^2)
    if(is.na(gradient[i,]) || (i>1 & all(abs(gradient[i,])< changes))){
      i=i-1
      break;
    }
  }
  return(list("i"=i, "p"=p, "g"=gradient, "loss"=loss))
}

data("marketing", package = "datarium")
head(marketing, 4)
x_1<-scale(marketing$youtube)
x_2<-scale(marketing$facebook)
y<-scale(marketing$sales)
p0<-c(0, 1)

#you can try difference step.size 
#you can try with unscale data with learning rate of 0.00002
l<-gd.lm(x_1, x_2, y, p0, max.iter = 300, step.size = 0.02)

(i = l[["i"]])
p = l[["p"]]
(p_last = p[i,])
l[["g"]][i,]
loss = l[['loss']]
g = l[['g']]

library(plotly)

x_axis = seq(0, 1, 0.01)
y_axis = seq(0, 1, 0.01)
n = length(x_axis)
z_axis = matrix(rep(0, len=n*n), nrow = n)

for (i in 1:n) {
  for (j in 1:n) {
    y_hat = x_axis[i] * x_1 + y_axis[j] * x_2
    z_axis[j,i] = mean((y_hat - y) ^ 2)
  }
}


fig <- plot_ly(x = x_axis, y = y_axis, z = z_axis) %>% add_surface()

# x y and z are different to what I show in collobaration session, 
# I noticed I should plot p[i, i] againt loss[i+1], otherwise the surface and trace will always be fitted.
fig <- fig %>% add_surface() %>%
  add_trace(x = p[1:n-1,1], y = p[1:n-1,2], z = loss[2:n], 
            type="scatter3d", mode="lines",
            line = list(color = "red", width = 4))
fig
