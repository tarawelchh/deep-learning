##### TASK B ###########
library(keras3)
library(tensorflow)
library(reticulate)

set.seed(123)
tensorflow::set_random_seed(123) ## practical 5 14-19
x_test <- read.csv("data/x_test.csv", header = FALSE)
xy_train <- read.csv("data/xy_train.csv", header = FALSE)
x_gen_train <- as.matrix(x_test[, 2:785]) # 10000 test images
x_gen_train2 <- as.matrix(xy_train[, 2:785]) # 500 train images
x_gen_all <- rbind(x_gen_train, x_gen_train2)
x_gen_all <- x_gen_all / 255

# Practical 5 lines 50-55
original_dim <- 784L
latent_dim <- 10L
intermediate_dim <- 256L
batch_size <- 128

# Directly from practical5_functions.R
layer_sampler <- new_layer_class(
  classname = "Sampler",
  call = function(z_mean, z_log_var) {
    epsilon <- tf$random$normal(shape = tf$shape(z_mean))
    z_mean + exp(0.5 * z_log_var) * epsilon
  }
)


# ENCODER: Practical 5 lines 63-72
encoder_inputs <- layer_input(shape = 28 * 28)

x <- encoder_inputs %>%
  layer_dense(intermediate_dim, activation = "relu")

z_mean <- x %>% layer_dense(latent_dim, name = "z_mean")
z_log_var <- x %>% layer_dense(latent_dim, name = "z_log_var")

encoder <- keras_model(encoder_inputs, list(z_mean, z_log_var),
  name = "encoder"
)

# DECODER: Practical 5 lines 82-89
latent_inputs <- layer_input(shape = c(latent_dim))

decoder_outputs <- latent_inputs %>%
  layer_dense(intermediate_dim, activation = "relu") %>%
  layer_dense(original_dim, activation = "sigmoid")

decoder <- keras_model(latent_inputs, decoder_outputs,
  name = "decoder"
)

# VAE Model class
# Practical 5 lines 97-148 — copied directly
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
# Practical 5 lines 158-161
vae <- model_vae(encoder, decoder)
vae %>% compile(optimizer = optimizer_adam())
vae %>% fit(x_gen_all, epochs = 50, shuffle = TRUE)

### GENERATE
# Sample 1000 random points from the latent space
# The KL penalty forces the latent space toward standard normal
# So we sample from N(0,1)

# Encode all real training images to find where they sit in latent space
encoded <- predict(encoder, x_gen_all)
z_means <- encoded[[1]] # the means

# Check the range of the encoded data
summary(z_means)

n_generate <- 1000
idx <- sample(1:nrow(x_gen_all), n_generate, replace = TRUE)
z_samples_near <- z_means[idx, ] + matrix(rnorm(n_generate * latent_dim, sd = 0.5),
  nrow = n_generate
)


z_samples_random <- matrix(rnorm(1000 * latent_dim), nrow = 1000, ncol = latent_dim)
#### SAMPLE TYPE ####
z_samples <- z_samples_random

# Decode into images — adapted from Practical 5 line 217
generated <- predict(decoder, z_samples)
# Convert back to 0-255 range
generated <- round(generated * 255)
generated[generated < 0] <- 0
generated[generated > 255] <- 255

# Check a few
par(mfrow = c(1, 3))
for (i in 50:54) {
  img <- matrix(generated[i, ], 28, 28)
  img <- img[1:28, 28:1]
  image(img, col = grey.colors(256), xaxt = "n", yaxt = "n")
}

encoded_2d <- predict(encoder, x_gen_all)
z_means_2d <- encoded_2d[[1]]
# Use labelled data only for colouring
x_labelled <- as.matrix(xy_train[, 2:785]) / 255
encoded_lab <- predict(encoder, x_labelled)
z_lab <- encoded_lab[[1]]

latent_dim <- 2L

# Rebuild encoder
encoder_inputs <- layer_input(shape = 28 * 28)
x <- encoder_inputs %>%
  layer_dense(intermediate_dim, activation = "relu")
z_mean <- x %>% layer_dense(latent_dim, name = "z_mean")
z_log_var <- x %>% layer_dense(latent_dim, name = "z_log_var")
encoder_2d <- keras_model(encoder_inputs, list(z_mean, z_log_var), name = "encoder_2d")

# Rebuild decoder
latent_inputs <- layer_input(shape = c(latent_dim))
decoder_outputs <- latent_inputs %>%
  layer_dense(intermediate_dim, activation = "relu") %>%
  layer_dense(original_dim, activation = "sigmoid")
decoder_2d <- keras_model(latent_inputs, decoder_outputs, name = "decoder_2d")

# Train
vae_2d <- model_vae(encoder_2d, decoder_2d)
vae_2d %>% compile(optimizer = optimizer_adam())
vae_2d %>% fit(x_gen_all, epochs = 50, shuffle = TRUE)


par(mfrow = c(1, 1))
plot(z_lab[, 1], z_lab[, 2],
  col = as.integer(factor(xy_train[, 1])),
  pch = 16, main = "2D VAE latent space", xlab = "Dim 1", ylab = "Dim 2"
)
legend("topright", legend = LETTERS[1:6], col = 1:6, pch = 16)

# Reshape generated images for the CNN
gen_cnn <- array_reshape(generated / 255, c(1000, 28, 28, 1))

# Use your trained classifier to predict
gen_pred_prob <- model_cnn %>% predict(gen_cnn)
gen_pred <- apply(gen_pred_prob, 1, which.max)
gen_labels <- LETTERS[gen_pred]

# How confident is the classifier?
gen_confidence <- apply(gen_pred_prob, 1, max)

# Results
table(gen_labels) # are all 6 classes represented?
cat("Mean confidence:", mean(gen_confidence), "\n") # higher = more realistic


# Pick 4 real images
test_idx <- c(1, 50, 100, 170)
test_images <- x_gen_all[test_idx, ]

# Reconstruct through 2D VAE (make sure 2D models are still loaded)
encoded_2d <- predict(encoder_2d, test_images)
z_2d <- encoded_2d[[1]] + exp(0.5 * encoded_2d[[2]]) * matrix(rnorm(4 * 2), nrow = 4)
recon_2d <- predict(decoder_2d, z_2d)

# Reconstruct through 10D VAE
encoded_10d <- predict(encoder_10d, test_images)
z_10d <- encoded_10d[[1]] + exp(0.5 * encoded_10d[[2]]) * matrix(rnorm(4 * 10), nrow = 4)
recon_10d <- predict(decoder_10d, z_10d)

# Show: original, 2D reconstruction, 10D reconstruction
par(mfrow = c(3, 4))
for (i in 1:4) {
  img <- matrix(test_images[i, ], 28, 28)
  image(img[1:28, 28:1], col = grey.colors(256), xaxt = "n", yaxt = "n", main = "Original")
}
for (i in 1:4) {
  img <- matrix(recon_2d[i, ], 28, 28)
  image(img[1:28, 28:1], col = grey.colors(256), xaxt = "n", yaxt = "n", main = "2D")
}
for (i in 1:4) {
  img <- matrix(recon_10d[i, ], 28, 28)
  image(img[1:28, 28:1], col = grey.colors(256), xaxt = "n", yaxt = "n", main = "10D")
}


#### GAN ####
x_gan <- x_gen_all # the combined training + test data we used for VAE
# Reshape to images
x_gan <- array_reshape(x_gan, c(nrow(x_gan), 28, 28, 1))
# Adapted from Practical 4 lines 270-273
latent_dim_gan <- 32
height <- 28
width <- 28
channels <- 1 # grayscale, not 3 (colour) as in Practical 4

# Adapted from Practical 4 lines 276-288
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

# Adapted from Practical 4 lines 295-316
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

# Practical 4 lines 320-330
# Replace the freeze_weights section with this:
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

# Check it worked — should show generator weights as trainable
summary(gan)

# Adapted from Practical 4 lines 333-394
iterations <- 2000
batch_size <- 32
start <- 1

for (step in 1:iterations) {
  # Generate fake images
  random_latent_vectors <- matrix(rnorm(batch_size * latent_dim_gan),
    nrow = batch_size, ncol = latent_dim_gan
  )
  generated_images <- generator %>% predict(random_latent_vectors, verbose = 0)

  # Get real images
  stop_idx <- start + batch_size - 1
  if (stop_idx > nrow(x_gan)) {
    start <- 1
    stop_idx <- batch_size
  }
  real_images <- x_gan[start:stop_idx, , , , drop = FALSE]

  # Combine and make labels
  rows <- nrow(real_images)
  combined_images <- array(0, dim = c(rows * 2, height, width, channels))
  combined_images[1:rows, , , ] <- generated_images
  combined_images[(rows + 1):(rows * 2), , , ] <- real_images
  labels <- rbind(
    matrix(1, nrow = batch_size, ncol = 1),
    matrix(0, nrow = batch_size, ncol = 1)
  )

  # Train discriminator
  d_loss <- discriminator %>% train_on_batch(combined_images, labels)

  # Train generator
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

    # Clamp values to [0,1]
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


# Find misclassified validation images (using augmented model)
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

# Visualise the misclassified ones with their probabilities
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

# Confidence per class on test set
for (l in LETTERS[1:6]) {
  idx <- which(pred_labels == l)
  conf <- apply(y_test_pred_prob[idx, ], 1, max)
  cat(l, ": mean confidence", round(mean(conf), 3), "\n")
}


# Encode the labelled training data
x_labelled <- as.matrix(xy_train[, 2:785]) / 255
encoded_labelled <- predict(encoder, x_labelled)
z_means <- encoded_labelled[[1]]

# Plot coloured by class
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

# Predict validation set with dense model too
pred_dense <- model %>% predict(x_val)
pred_dense_labels <- LETTERS[apply(pred_dense, 1, which.max)]

# Images all three models got wrong
wrong_dense <- (pred_dense_labels != true_labels)
wrong_noaug <- (pred_noaug_labels != true_labels)
wrong_aug <- (pred_aug_labels != true_labels)

# Wrong across multiple models
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


# Extract first conv layer weights
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

# Compare generated images to training images
# Using Euclidean distance between pixel vectors
x_all_real <- rbind(as.matrix(xy_train[, 2:785]), as.matrix(x_test[, 2:785]))

min_distances <- c()
closest_idx <- c()

for (i in 1:1000) {
  gen_img <- generated[i, ]
  # Distance to every real image
  dists <- apply(x_all_real, 1, function(real) sqrt(sum((gen_img - real)^2)))
  min_distances[i] <- min(dists)
  closest_idx[i] <- which.min(dists)
}

# Summary
cat("Mean min distance:", round(mean(min_distances), 2), "\n")
cat("Smallest min distance:", round(min(min_distances), 2), "\n")

# For comparison, what's the typical distance between two real images?
sample_pairs <- sample(1:nrow(x_all_real), 100)
real_dists <- c()
for (i in 1:50) {
  real_dists[i] <- sqrt(sum((x_all_real[sample_pairs[i], ] -
    x_all_real[sample_pairs[i + 50], ])^2))
}
cat("Mean distance between random real pairs:", round(mean(real_dists), 2), "\n")



# Random N(0,1) sampling
z_random <- matrix(rnorm(1000 * latent_dim), nrow = 1000, ncol = latent_dim)
gen_random <- predict(decoder, z_random)
gen_random <- round(gen_random * 255)
gen_random[gen_random < 0] <- 0
gen_random[gen_random > 255] <- 255

# Near-real-data sampling
encoded <- predict(encoder, x_gen_all)
z_means <- encoded[[1]]
idx <- sample(1:nrow(x_gen_all), 1000, replace = TRUE)
z_near <- z_means[idx, ] + matrix(rnorm(1000 * latent_dim, sd = 0.5), nrow = 1000)
gen_near <- predict(decoder, z_near)
gen_near <- round(gen_near * 255)
gen_near[gen_near < 0] <- 0
gen_near[gen_near > 255] <- 255

# Show side by side
par(mfrow = c(2, 4))
for (i in 1:4) {
  img <- matrix(gen_random[i, ], 28, 28)
  image(img[1:28, 28:1],
    col = grey.colors(256), xaxt = "n", yaxt = "n",
    main = "Random"
  )
}
for (i in 1:4) {
  img <- matrix(gen_near[i, ], 28, 28)
  image(img[1:28, 28:1],
    col = grey.colors(256), xaxt = "n", yaxt = "n",
    main = "Near-data"
  )
}
