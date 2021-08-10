## Step 1: Load the Boston housing dataset
library(keras)

dataset <- dataset_boston_housing()
c(c(train_data, train_targets), c(test_data, test_targets)) %<-% dataset

## Step 2: Perform feature-wise normalisation
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
train_data <- scale(train_data, center = mean, scale = std)
test_data <- scale(test_data, center = mean, scale = std)

# two hidden layers, each with 64 nodes
# in general overfitting is more of an issue with smaller datasets and
#  reducing the size of the network is one way to mitigate this

# network ends with a single unit and no activation (i.e. linear layer)
# typical setup for scalar regression (i.e. regression trying to predict a single
#  continuous value).
# applying an activation function constrains the range the output can take
build_model <- function() {
  model <- keras_model_sequential() %>% 
    layer_dense(units = 64, activation = "relu",
                input_shape = dim(train_data)[[2]]) %>% 
    layer_dense(units = 64, activation = "relu") %>% 
    layer_dense(units = 1)
  
  model %>% compile(
    optimizer = "rmsprop",
    loss = "mse", # mse loss function widely used for regression problems
    metrics = c("mae", "mape") # mae of 0.5 means predictions off by $500 by average here
  )
}

## Step 3: Validating approach using k-fold validation
k <- 2
indices <- sample(1:nrow(train_data))
folds <- cut(indices, breaks = k, labels = FALSE)
num_epochs <- 500
all_mae_histories <- NULL
all_mape_histories <- NULL

for (i in 1:k) {
  cat("processing fold #", i, "\n")
  
  # Prepare the validation data: data from partition # k
  val_indices <- which(folds == i, arr.ind = TRUE)
  val_data <- train_data[val_indices,]
  val_targets <- train_targets[val_indices]
  
  # Prepare the training data: data from all other partitions
  partial_train_data <- train_data[-val_indices,]
  partial_train_targets <- train_targets[-val_indices]
  
  # Build the Keras model (already compiled)
  model <- build_model()
  
  # Train the model (in silent mode, verbose=0)
  history <- model %>% fit(
    partial_train_data, partial_train_targets,
    validation_data = list(val_data, val_targets),
    epochs = num_epochs, batch_size = 1, verbose = 0
  )
  mae_history <- history$metrics$val_mae
  all_mae_histories <- rbind(all_mae_histories, mae_history)
  
  mape_history <- history$metrics$val_mape
  all_mape_histories <- rbind(all_mape_histories, mape_history)
}

# compute average of per-epoch MAE scores for all folds
average_mae_history <- data.frame(
  epoch = seq(1:ncol(all_mae_histories)),
  validation_mae = apply(all_mae_histories, 2, mean)
)

average_mape_history <- data.frame(
  epoch = seq(1:ncol(all_mape_histories)),
  validation_mape = apply(all_mape_histories, 2, mean)
)

# plot validation scores
library(ggplot2)
ggplot(average_mae_history, aes(x = epoch, y = validation_mae)) + geom_line()

ggplot(average_mae_history, aes(x = epoch, y = validation_mae)) + geom_smooth()

ggplot(average_mape_history, aes(x = epoch, y = validation_mape)) + geom_line()

ggplot(average_mape_history, aes(x = epoch, y = validation_mape)) + geom_smooth()
