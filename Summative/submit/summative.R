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
par(mfrow = c(3, 6))
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


for (i in 1:500) view_image(i)

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


###############
##### TASK B ###########
library(keras3)
library(tensorflow)
library(reticulate)

set.seed(123)
tensorflow::set_random_seed(123) ## practical 5 14-19

# Practical 5 used:
# x_train <- mnist$train$x/255
# x_train <- array_reshape(x_train, c(nrow(x_train), 28*28), order = "F")

# combine train and test images for more training data
x_gen_train <- as.matrix(x_test[, 2:785]) # 10000 test images
x_gen_train2 <- as.matrix(xy_train[, 2:785]) # 500 train images
x_gen_all <- rbind(x_gen_train, x_gen_train2)
x_gen_all <- x_gen_all / 255

original_dim <- 784L
latent_dim <- 10L
intermediate_dim <- 256L
batch_size <- 128

layer_sampler <- new_layer_class(
  classname = "Sampler",
  call = function(z_mean, z_log_var) {
    epsilon <- tf$random$normal(shape = tf$shape(z_mean))
    z_mean + exp(0.5 * z_log_var) * epsilon
  }
)

encoder_inputs <- layer_input(shape = 28 * 28)

x <- encoder_inputs %>%
  layer_dense(intermediate_dim, activation = "relu")

z_mean <- x %>% layer_dense(latent_dim, name = "z_mean")
z_log_var <- x %>% layer_dense(latent_dim, name = "z_log_var")

encoder <- keras_model(encoder_inputs, list(z_mean, z_log_var),
  name = "encoder"
)

latent_inputs <- layer_input(shape = c(latent_dim))

decoder_outputs <- latent_inputs %>%
  layer_dense(intermediate_dim, activation = "relu") %>%
  layer_dense(original_dim, activation = "sigmoid")

decoder <- keras_model(latent_inputs, decoder_outputs,
  name = "decoder"
)

# VAE Model class
model_vae <- new_model_class(
  classname = "VAE",
  initialize = function(encoder, decoder, ...) {
    super$initialize(...)
    self$encoder <- encoder
    self$decoder <- decoder
    self$sampler <- layer_sampler()
    self$total_loss_tracker <-
      metric_mean(name = "total_loss")
    self$reconstruction_loss_tracker <-
      metric_mean(name = "reconstruction_loss")
    self$kl_loss_tracker <-
      metric_mean(name = "kl_loss")
  },
  metrics = mark_active(function() {
    list(
      self$total_loss_tracker,
      self$reconstruction_loss_tracker,
      self$kl_loss_tracker
    )
  }),
  train_step = function(data) {
    with(tf$GradientTape() %as% tape, {
      c(z_mean, z_log_var) %<-% self$encoder(data)
      z <- self$sampler(z_mean, z_log_var)

      reconstruction <- decoder(z)
      reconstruction_loss <-
        loss_binary_crossentropy(data, reconstruction) %>%
        sum(axis = c(1)) %>%
        mean()

      kl_loss <- -0.5 * (1 + z_log_var - z_mean^2 - exp(z_log_var))
      total_loss <- reconstruction_loss + mean(kl_loss)
    })

    grads <- tape$gradient(total_loss, self$trainable_weights)
    self$optimizer$apply_gradients(zip_lists(grads, self$trainable_weights))

    self$total_loss_tracker$update_state(total_loss)
    self$reconstruction_loss_tracker$update_state(reconstruction_loss)
    self$kl_loss_tracker$update_state(kl_loss)

    list(
      total_loss = self$total_loss_tracker$result(),
      reconstruction_loss = self$reconstruction_loss_tracker$result(),
      kl_loss = self$kl_loss_tracker$result()
    )
  }
)

## TRAIN
vae <- model_vae(encoder, decoder)
vae %>% compile(optimizer = optimizer_adam())
vae %>% fit(x_gen_all, epochs = 50, shuffle = TRUE)

### GENERATE
# Sample 1000 random points from the latent space
# KL penalty forces the latent space toward standard normal
# So we sample from N(0,1)

# encode all real training images to find where they sit in latent space
encoded <- predict(encoder, x_gen_all)
z_means <- encoded[[1]] # the means

# check the range of the encoded data
summary(z_means)

n_generate <- 1000
idx <- sample(1:nrow(x_gen_all), n_generate, replace = TRUE)
z_samples_near <- z_means[idx, ] + matrix(rnorm(n_generate * latent_dim, sd = 0.3),
  nrow = n_generate
)

z_samples_random <- matrix(rnorm(1000 * latent_dim), nrow = 1000, ncol = latent_dim)

z_samples <- z_samples_near
# decode into images
generated <- predict(decoder, z_samples)
# convert back to 0-255 range
generated <- round(generated * 255)
generated[generated < 0] <- 0
generated[generated > 255] <- 255

# check a few
par(mfrow = c(1, 3))
for (i in 5:500) {
  img <- matrix(generated[i, ], 28, 28)
  img <- img[1:28, 28:1]
  image(img, col = grey.colors(256), xaxt = "n", yaxt = "n")
}

x_labelled <- as.matrix(xy_train[, 2:785]) / 255
encoded_lab <- predict(encoder, x_labelled)
z_lab <- encoded_lab[[1]]

par(mfrow = c(1, 1))
plot(z_lab[, 1], z_lab[, 2],
  col = as.integer(factor(xy_train[, 1])),
  pch = 16, main = "2D VAE latent space", xlab = "Dim 1", ylab = "Dim 2"
)
legend("topright", legend = LETTERS[1:6], col = 1:6, pch = 16)

# reshape generated images for the CNN
gen_cnn <- array_reshape(generated / 255, c(1000, 28, 28, 1))

# use your trained classifier to predict
gen_pred_prob <- model_cnn %>% predict(gen_cnn)
gen_pred <- apply(gen_pred_prob, 1, which.max)
gen_labels <- LETTERS[gen_pred]
gen_confidence <- apply(gen_pred_prob, 1, max)

# results
table(gen_labels) # are all 6 classes represented?
cat("Mean confidence:", mean(gen_confidence), "\n") # higher = more realistic
# pick 4 real images
test_idx <- c(1, 50, 100, 150)
test_images <- x_gen_all[test_idx, ]

write.table(generated, "new.csv",
  row.names = FALSE,
  col.names = FALSE, sep = ","
)

check <- read.csv("new.csv", header = FALSE)
cat("Dimensions:", dim(check), "\n") # should be 1000 x 784

for (i in 1:1000) {
  view_image(i, dataset = check, label_col = FALSE)
}

#### GAN ####
x_gan <- x_gen_all # the combined training + test data we used for VAE
x_gan <- array_reshape(x_gan, c(nrow(x_gan), 28, 28, 1))
latent_dim_gan <- 32
height <- 28
width <- 28
channels <- 1 # grayscale, not 3 (colour)

generator_input <- layer_input(shape = c(latent_dim_gan))
generator_output <- generator_input %>%
  layer_dense(units = 128 * 7 * 7) %>%
  layer_activation_leaky_relu() %>%
  layer_reshape(target_shape = c(7, 7, 128)) %>%
  layer_conv_2d_transpose(
    filters = 128, kernel_size = 4,
    strides = 2, padding = "same"
  ) %>%
  layer_activation_leaky_relu() %>%
  layer_conv_2d_transpose(
    filters = 64, kernel_size = 4,
    strides = 2, padding = "same"
  ) %>%
  layer_activation_leaky_relu() %>%
  layer_conv_2d(
    filters = channels, kernel_size = 7,
    activation = "sigmoid", padding = "same"
  )
generator <- keras_model(generator_input, generator_output)

discriminator_input <- layer_input(shape = c(height, width, channels))
discriminator_output <- discriminator_input %>%
  layer_conv_2d(filters = 64, kernel_size = 3) %>%
  layer_activation_leaky_relu() %>%
  layer_conv_2d(filters = 128, kernel_size = 4, strides = 2) %>%
  layer_activation_leaky_relu() %>%
  layer_conv_2d(filters = 128, kernel_size = 4, strides = 2) %>%
  layer_activation_leaky_relu() %>%
  layer_flatten() %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 1, activation = "sigmoid")
discriminator <- keras_model(discriminator_input, discriminator_output)

discriminator_optimizer <- optimizer_rmsprop(
  learning_rate = 0.0001,
  clipvalue = 1.0
)
discriminator %>% compile(
  optimizer = discriminator_optimizer,
  loss = "binary_crossentropy"
)

discriminator$trainable <- FALSE

gan_input <- layer_input(shape = c(latent_dim_gan))
gan_output <- discriminator(generator(gan_input))
gan <- keras_model(gan_input, gan_output)

gan_optimizer <- optimizer_rmsprop(
  learning_rate = 0.0003,
  clipvalue = 1.0
)
gan %>% compile(
  optimizer = gan_optimizer,
  loss = "binary_crossentropy"
)
summary(gan)

loss <- data.frame(dloss = numeric(25), gloss = numeric(25))
iterations <- 2500
batch_size <- 32
start <- 1

for (step in 1:iterations) {
  # generate fake images
  random_latent_vectors <- matrix(rnorm(batch_size * latent_dim_gan),
    nrow = batch_size, ncol = latent_dim_gan
  )
  generated_images <- generator %>% predict(random_latent_vectors, verbose = 0)

  # get real images
  stop_idx <- start + batch_size - 1
  if (stop_idx > nrow(x_gan)) {
    start <- 1
    stop_idx <- batch_size
  }
  real_images <- x_gan[start:stop_idx, , , , drop = FALSE]

  # combine and make labels
  rows <- nrow(real_images)
  combined_images <- array(0, dim = c(rows * 2, height, width, channels))
  combined_images[1:rows, , , ] <- generated_images
  combined_images[(rows + 1):(rows * 2), , , ] <- real_images
  labels <- rbind(
    matrix(1, nrow = batch_size, ncol = 1),
    matrix(0, nrow = batch_size, ncol = 1)
  )

  # train discriminator
  d_loss <- discriminator %>% train_on_batch(combined_images, labels)

  # train generator
  random_latent_vectors <- matrix(rnorm(batch_size * latent_dim_gan),
    nrow = batch_size, ncol = latent_dim_gan
  )
  misleading_targets <- array(0, dim = c(batch_size, 1))
  a_loss <- gan %>% train_on_batch(random_latent_vectors, misleading_targets)

  start <- start + batch_size

  if (step %% 100 == 0) {
    cat(
      "Step:", step,
      " D loss:", d_loss,
      " G loss:", a_loss, "\n"
    )

    loss[step %/% 100, "dloss"] <- d_loss
    loss[step %/% 100, "gloss"] <- a_loss
    # clamp values to [0,1]
    show_real <- pmin(pmax(real_images[1, , , 1], 0), 1)
    show_gen <- pmin(pmax(generated_images[1, , , 1], 0), 1)

    par(mfrow = c(1, 2))
    plot(0,
      type = "n", xlim = c(0, 1), ylim = c(0, 1),
      xaxt = "n", yaxt = "n", main = "Real"
    )
    rasterImage(show_real, 0, 0, 1, 1)
    plot(0,
      type = "n", xlim = c(0, 1), ylim = c(0, 1),
      xaxt = "n", yaxt = "n", main = "Generated"
    )
    rasterImage(show_gen, 0, 0, 1, 1)
  }
}


# find misclassified validation images (using augmented model)
wrong_idx <- which(pred_aug_labels != true_labels)

cat("Misclassified images:\n")
for (i in wrong_idx) {
  probs <- round(pred_aug[i, ], 3)
  names(probs) <- LETTERS[1:6]
  cat(
    "True:", true_labels[i],
    " Predicted:", pred_aug_labels[i],
    " Probabilities:", paste(names(probs), probs, collapse = ", "), "\n"
  )
}

# visualise the misclassified ones with their probabilities
par(mfrow = c(2, min(4, length(wrong_idx))))
for (i in wrong_idx) {
  pixels <- x_val_cnn[i, , , 1]
  img <- pixels[1:28, 28:1]
  top_prob <- round(max(pred_aug[i, ]), 2)
  image(img,
    col = grey.colors(256), xaxt = "n", yaxt = "n",
    main = paste(
      "True:", true_labels[i],
      "\nPred:", pred_aug_labels[i],
      "(", top_prob, ")"
    )
  )
}

# confidence per class on test set
for (l in LETTERS[1:6]) {
  idx <- which(pred_labels == l)
  conf <- apply(y_test_pred_prob[idx, ], 1, max)
  cat(l, ": mean confidence", round(mean(conf), 3), "\n")
}


# encode the labelled training data
x_labelled <- as.matrix(xy_train[, 2:785]) / 255
encoded_labelled <- predict(encoder, x_labelled)
z_means <- encoded_labelled[[1]]

plot(z_means[, 1], z_means[, 2],
  col = as.integer(factor(xy_train[, 1])),
  pch = 16, cex = 0.8,
  xlab = "Latent dim 1", ylab = "Latent dim 2",
  main = "VAE latent space by class"
)
legend("topright",
  legend = LETTERS[1:6],
  col = 1:6, pch = 16, bty = "n"
)

# predict validation set with dense model too
pred_dense <- model %>% predict(x_val)
pred_dense_labels <- LETTERS[apply(pred_dense, 1, which.max)]

# all three models got wrong
wrong_dense <- (pred_dense_labels != true_labels)
wrong_noaug <- (pred_noaug_labels != true_labels)
wrong_aug <- (pred_aug_labels != true_labels)

# erong across multiple models
hard_images <- which(wrong_aug & wrong_noaug)
cat("Hard for both dense and CNN-no-aug:", length(hard_images), "\n")

always_wrong <- which(wrong_dense & wrong_noaug & wrong_aug)
cat("Hard for ALL models:", length(always_wrong), "\n")


pc <- prcomp(z_means)
plot(pc$x[, 1], pc$x[, 2],
  col = as.integer(factor(xy_train[, 1])),
  pch = 16, cex = 0.8,
  xlab = "PC1", ylab = "PC2",
  main = "PCA of VAE latent space"
)
legend("topright",
  legend = LETTERS[1:6],
  col = 1:6, pch = 16, bty = "n"
)


# extract first conv layer weights
weights <- get_weights(model_cnn)[[1]] # first layer kernels
par(mfrow = c(4, 8))
for (i in 1:32) {
  kernel <- weights[, , 1, i]
  image(kernel,
    col = grey.colors(256), xaxt = "n", yaxt = "n",
    main = paste(i)
  )
}

confidence <- apply(y_test_pred_prob, 1, max)
hardest <- order(confidence)[1:8]

par(mfrow = c(2, 4))
for (i in hardest) {
  pixels <- x_test_cnn[i, , , 1]
  img <- pixels[1:28, 28:1]
  conf <- round(confidence[i], 2)
  image(img,
    col = grey.colors(256), xaxt = "n", yaxt = "n",
    main = paste(pred_labels[i], "(", conf, ")")
  )
}

# Euclidean distance between pixel vectors
x_all_real <- rbind(as.matrix(xy_train[, 2:785]), as.matrix(x_test[, 2:785]))

min_distances <- c()
closest_idx <- c()

for (i in 1:1000) {
  gen_img <- generated[i, ]
  # distance to every real image
  dists <- apply(x_all_real, 1, function(real) sqrt(sum((gen_img - real)^2)))
  min_distances[i] <- min(dists)
  closest_idx[i] <- which.min(dists)
}

cat("Mean min distance:", round(mean(min_distances), 2), "\n")
cat("Smallest min distance:", round(min(min_distances), 2), "\n")

sample_pairs <- sample(1:nrow(x_all_real), 100)
real_dists <- c()
for (i in 1:50) {
  real_dists[i] <- sqrt(sum((x_all_real[sample_pairs[i], ] -
    x_all_real[sample_pairs[i + 50], ])^2))
}
cat("Mean distance between random real pairs:", round(mean(real_dists), 2), "\n")


# random N(0,1) sampling
z_random <- matrix(rnorm(1000 * latent_dim), nrow = 1000, ncol = latent_dim)
gen_random <- predict(decoder, z_random)
gen_random <- round(gen_random * 255)
gen_random[gen_random < 0] <- 0
gen_random[gen_random > 255] <- 255

# near-real-data sampling
encoded <- predict(encoder, x_gen_all)
z_means <- encoded[[1]]
idx <- sample(1:nrow(x_gen_all), 1000, replace = TRUE)
z_near <- z_means[idx, ] + matrix(rnorm(1000 * latent_dim, sd = 0.5), nrow = 1000)
gen_near <- predict(decoder, z_near)
gen_near <- round(gen_near * 255)
gen_near[gen_near < 0] <- 0
gen_near[gen_near > 255] <- 255

# side by side
par(mfrow = c(2, 4))
for (i in 10:14) {
  img <- matrix(gen_random[i, ], 28, 28)
  image(img[1:28, 28:1],
    col = grey.colors(256), xaxt = "n", yaxt = "n",
    main = "Random"
  )
}
for (i in 10:14) {
  img <- matrix(gen_near[i, ], 28, 28)
  image(img[1:28, 28:1],
    col = grey.colors(256), xaxt = "n", yaxt = "n",
    main = "Near-data"
  )
}


x_labelled <- as.matrix(xy_train[, 2:785]) / 255
encoded_labelled <- predict(encoder, x_labelled)
z_means <- encoded_labelled[[1]]

# PCA to reduce 10D to 2D
pc <- prcomp(z_means)
plot(pc$x[, 1], pc$x[, 2],
  col = as.integer(factor(xy_train[, 1])),
  pch = 16, cex = 0.8,
  xlab = "PC1", ylab = "PC2",
  main = "PCA of 10D VAE latent space"
)
legend("topright",
  legend = LETTERS[1:6],
  col = 1:6, pch = 16, bty = "n"
)

# for each generated image, find its closest real image
x_all_real <- rbind(as.matrix(xy_train[, 2:785]), as.matrix(x_test[, 2:785]))

min_distances <- c()
for (i in 1:1000) {
  gen_img <- generated[i, ]
  dists <- apply(x_all_real, 1, function(real) sqrt(sum((gen_img - real)^2)))
  min_distances[i] <- min(dists)
}

cat("Mean distance to nearest real image:", round(mean(min_distances), 1), "\n")
cat("Min distance (most suspicious):", round(min(min_distances), 1), "\n")

# dist between random pairs of real images
real_dists <- c()
for (i in 1:500) {
  pair <- sample(1:nrow(x_all_real), 2)
  real_dists[i] <- sqrt(sum((x_all_real[pair[1], ] - x_all_real[pair[2], ])^2))
}

cat("Mean distance between real pairs:", round(mean(real_dists), 1), "\n")

d_losses <- loss[, "dloss"]
g_losses <- loss[, "gloss"]

plot(seq(100, 2500, 100), d_losses,
  type = "b",
  ylim = c(0.5, 0.8), xlab = "Iteration", ylab = "Loss",
  col = "red", pch = 16
)
lines(seq(100, 2500, 100), g_losses,
  type = "b",
  col = "blue", pch = 16
)
legend("right",
  legend = c("Discriminator", "Generator"),
  col = c("red", "blue"), pch = 16
)
