library(keras)
dataset <- dataset_boston_housing()
train_data <- dataset$train$x
train_y <- dataset$train$y

test_data <- dataset$test$x
test_y <- dataset$test$y


mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
train_data <- scale(train_data, center = mean, scale = std)
test_data <- scale(test_data, center = mean, scale = std)


model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu", input_shape = c(ncol(test_data))) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 1)


model %>% compile(
  optimizer = "adam",
  loss = "mse",
  metrics = c("mae")
)

ModelHistory <- model %>% fit(train_data, train_y, epochs = 300, batch_size = 8, validation_split=1/3)


library(ggplot2)

PlotData <- data.frame(x = c(5:ModelHistory$params$epochs), y = ModelHistory$metrics$val_mae [-c(1:4)])
ggplot(PlotData, aes(x = x, y = y)) + geom_smooth() + xlab("Epoch") + ylab("Estimated Validation MAE loss")

------------------------
  
model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu", input_shape = c(ncol(test_data))) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 1)

model %>% compile(
  optimizer = "adam",
  loss = "mse",
  metrics = c("mae")
)

model %>% fit(train_data, train_y, epochs = 91, batch_size = 8, validation_data = list(test_data, test_y))

result <- model %>% evaluate(test_data, test_y)