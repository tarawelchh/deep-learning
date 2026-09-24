library(keras3)
library(dplyr)
set.seed(123)
tensorflow::set_random_seed(123)

######## load the data ########
xy_train <- read.csv("data/xy_train.csv", header = FALSE)
x_test <- read.csv("data/x_test.csv", header = FALSE)

dim(xy_train) # 500 x 785
table(xy_train[, 1]) # labels A,B,C,D,E,F
dim(x_test) # 10k x 785
###########

#### view images ######
par(mfrow = c(4, 3))
view_image <- function(row, dataset = xy_train, label_col = TRUE) {
  if (label_col) {
    pixels <- as.numeric(dataset[row, 2:785])
    title_text <- (dataset[row, 1])
  } else {
    pixels <- as.numeric(dataset[row, 1:784])
    title_text <- ""
  }
  img <- matrix(pixels, 28, 28)
  img <- img[1:28, 28:1]
  image(img, col = grey.colors(256), main = title_text, xaxt = "n", yaxt = "n")
}


for (i in 1:10) view_image(i)

par(mfrow = c(1, 6), mar = c(1, 1, 2, 1))
for (label in LETTERS[1:6]) {
  idx <- which(xy_train[, 1] == label)[7]
  pixels <- as.numeric(xy_train[idx, 2:785])
  img <- matrix(pixels, 28, 28)
  image(img[1:28, 28:1],
    col = grey.colors(256),
    xaxt = "n", yaxt = "n", main = label, cex.main = 3
  )
}

#########

###### data prep #######
labels <- xy_train[, 1] # first column = labels (A, B, C, D, E, F)
x_train_all <- as.matrix(xy_train[, 2:785]) # remaining 784 columns = pixels
x_train_all <- x_train_all / 255 # normalise to 0-1

label_numbers <- as.integer(factor(labels)) - 1 # A=0, B=1, ..., F=5
y_train_all <- to_categorical(label_numbers, num_classes = 6) # one hot encode

val_idx <- sample(1:500, 100) # 20 percent val, 80 train
train_idx <- setdiff(1:500, val_idx)

x_train <- x_train_all[train_idx, ]
y_train <- y_train_all[train_idx, ]
x_val <- x_train_all[val_idx, ]
y_val <- y_train_all[val_idx, ]

#########


######## first try model train ########
input <- layer_input(shape = c(784L))
output <- input %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 6, activation = "softmax")
model <- keras_model(input, output)

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c("accuracy")
)

history <- model %>% fit(
  x_train, y_train,
  epochs = 50,
  batch_size = 32,
  validation_data = list(x_val, y_val)
)

plot(history)

model %>% evaluate(x_val, y_val)
###########

####### param tuning #########
neurons <- c(256, 128, 64, 32)
dropout <- seq(0.1, 0.4, 0.1)
grid <- expand.grid(n1 = neurons, n2 = neurons, d = dropout)
grid <- data.frame(grid)
# Store results
results <- data.frame(
  n1 = integer(), n2 = integer(),
  d = numeric(), val_accuracy = numeric()
)

grid <- grid[grid$n1 >= grid$n2, ]

for (i in 1:nrow(grid)) {
  cfg <- grid[i, ]
  input <- layer_input(shape = c(784))
  output <- input %>%
    layer_dense(units = cfg$n1, activation = "relu") %>%
    layer_dropout(rate = cfg$d) %>%
    layer_dense(units = cfg$n2, activation = "relu") %>%
    layer_dropout(rate = cfg$d) %>%
    layer_dense(units = 6, activation = "softmax")
  model <- keras_model(input, output)

  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_rmsprop(),
    metrics = c("accuracy")
  )

  # train
  model %>% fit(
    x_train, y_train,
    epochs = 50,
    batch_size = 32,
    validation_data = list(x_val, y_val),
    verbose = 0
  )

  score <- model %>% evaluate(x_val, y_val, verbose = 0)
  val_acc <- score$accuracy # first element is loss, second is accuracy

  cat(
    "Units:", cfg$n1, "->", cfg$n2,
    " Dropout:", cfg$d,
    " Val accuracy:", round(val_acc, 3), "\n"
  )

  results <- rbind(results, data.frame(
    n1 = cfg$n1, n2 = cfg$n2,
    d = cfg$d, val_accuracy = val_acc
  ))
}

print(results)
print(which.max(results$val_accuracy))
## 256->128, 0.1 dropout each time has accuracy 0.78

input <- layer_input(shape = c(784))
output <- input %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 6, activation = "softmax")
model <- keras_model(input, output)

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c("accuracy")
)

history <- model %>% fit(
  x_train, y_train,
  epochs = 50,
  batch_size = 32,
  validation_data = list(x_val, y_val)
)

plot(history)
model %>% evaluate(x_val, y_val)

#### CNN ######

# reshape from flat 784 to 28x28x1 images
x_train_cnn <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1))
x_val_cnn <- array_reshape(x_val, c(nrow(x_val), 28, 28, 1))

# build CNN- P3 used 32->64->128->128 filters on 150x150 images
# scale down here for 28x28 images
input <- layer_input(shape = c(28, 28, 1))
output <- input %>%
  # convolutional layers
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  # flatten and classify
  layer_flatten() %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 6, activation = "softmax")
model_cnn_no_aug <- keras_model(input, output)


model_cnn_no_aug %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c("accuracy")
)

# train
history_cnn_no_aug <- model_cnn_no_aug %>% fit(
  x_train_cnn, y_train,
  epochs = 50,
  batch_size = 32,
  validation_data = list(x_val_cnn, y_val)
)

plot(history_cnn_no_aug)
model_cnn_no_aug %>% evaluate(x_val_cnn, y_val)
# 0.88 acuracy
#######


#### CNN with aug #####
data_augmentation <- keras_model_sequential(input_shape = c(28, 28, 1)) %>%
  layer_random_rotation(factor = 0.05) %>%
  layer_random_flip("horizontal") %>%
  layer_random_flip("vertical") %>%
  layer_random_translation(height_factor = 0.1, width_factor = 0.1)
#####

input <- layer_input(shape = c(28, 28, 1))
output <- input %>%
  data_augmentation() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 6, activation = "softmax")
model_cnn <- keras_model(input, output)

model_cnn %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c("accuracy")
)

# train — more epochs since augmentation slows learning
# also add early stopping so we don't overtrain
history_cnn <- model_cnn %>% fit(
  x_train_cnn, y_train,
  epochs = 100,
  batch_size = 32,
  validation_data = list(x_val_cnn, y_val),
  callbacks = list(
    callback_terminate_on_nan(),
    callback_early_stopping(
      monitor = "val_loss", patience = 10,
      restore_best_weights = TRUE
    )
  )
)

plot(history_cnn)
model_cnn %>% evaluate(x_val_cnn, y_val)

#### compare aug vs no aug #####
true_labels <- LETTERS[apply(y_val, 1, which.max)]

# CNN WITHOUT augmentation
pred_noaug <- model_cnn_no_aug %>% predict(x_val_cnn)
pred_noaug_labels <- LETTERS[apply(pred_noaug, 1, which.max)]

# CNN WITH augmentation
pred_aug <- model_cnn %>% predict(x_val_cnn)
pred_aug_labels <- LETTERS[apply(pred_aug, 1, which.max)]

cat("WITHOUT augmentation:\n")
for (l in LETTERS[1:6]) {
  idx <- which(true_labels == l)
  if (length(idx) > 0) {
    acc <- mean(pred_noaug_labels[idx] == l)
    cat(l, ":", round(acc, 3), "(", sum(true_labels == l), "samples)\n")
  }
}

cat("\nWITH augmentation:\n")
for (l in LETTERS[1:6]) {
  idx <- which(true_labels == l)
  if (length(idx) > 0) {
    acc <- mean(pred_aug_labels[idx] == l)
    cat(l, ":", round(acc, 3), "(", sum(true_labels == l), "samples)\n")
  }
}

# confusion matrices
cat("\nConfusion matrix WITHOUT augmentation:\n")
print(table(Predicted = pred_noaug_labels, True = true_labels))

cat("\nConfusion matrix WITH augmentation:\n")
print(table(Predicted = pred_aug_labels, True = true_labels))


#########

#### predictions ######
x_test_data <- as.matrix(x_test[, 2:785])
x_test_data <- x_test_data / 255
x_test_cnn <- array_reshape(x_test_data, c(nrow(x_test_data), 28, 28, 1))

y_test_pred_prob <- model_cnn %>% predict(x_test_cnn)
y_test_pred <- apply(y_test_pred_prob, 1, which.max)

# convert back to letters (1=A, 2=B, etc.)
pred_labels <- LETTERS[y_test_pred]
table(pred_labels)

predictions <- x_test
predictions[, 1] <- pred_labels
write.table(predictions, "output/predictions.csv",
  row.names = FALSE,
  col.names = FALSE, sep = ","
)
check <- read.csv("output/predictions.csv", header = FALSE)
dim(check) # should be 10000 x 785
table(check[, 1]) # should match pred_labels

par(mfrow = c(3, 6))
for (i in sample(1:10000, 1000)) {
  pixels <- x_test_cnn[i, , , 1]
  img <- pixels[1:28, 28:1]
  image(img,
    col = grey.colors(256), xaxt = "n", yaxt = "n",
    main = paste("Predicted:", pred_labels[i])
  )
}
