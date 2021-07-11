## Step 1: Load the data
library(keras)

# keeping only the top 5,000 most frequently occurring words in the training data
imdb <- dataset_imdb(num_words = 10000)
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% imdb

## Step 2: Turns lists into tensors
# one approach is to one-hot encode the lists then use a dense layer capable of handling floating-point vector data as
#  the first layer. The other approach (not shown here) is to pad the lists, turn them into an integer
#  tensor of shape and then use as the first layer a layer capable of handling such a tensor
vectorise_sequences <- function(sequences, dimension = 10000) {
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  for (i in 1:length(sequences))
    results[i, sequences[[i]]] <- 1
  results
}

x_train <- vectorise_sequences(train_data)
y_train <- vectorise_sequences(test_data)

## Step 3: Convert labels from integer to numeric
y_train <- as.numeric(train_labels)
y_test <- as.numeric(test_labels)

## Step 4: Build the model
model <- keras_model_sequential() %>% 
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>% 
  layer_dense(units = 16, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

## Step 5: Choose loss function and optimiser
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

## Step 6: Create a validation set
val_indices <- 1:2000
x_val <- x_train[val_indices,]
partial_x_train <- x_train[-val_indices,]
y_val <- y_train[val_indices]
partial_y_train <- y_train[-val_indices]

## Step 7: Train the model
history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)

## Step 8: Visualise training and validation metrics
plot(history)

## Step 9: Generate predictions on new data