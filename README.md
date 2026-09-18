# Image Classification and Generation
## Deep Learning and Artificial Intelligence Module (4th Year)
We perform classification of 10,000 images based on a training set of 500 labelled images. We then aim to generate
1000 new images similar to those within the dataset.

The data consists of 6 different categories of 28×28 pixel images, in which A and B are circle-based, C and D are
line-based and E and F resemble a D shape.

## Classification 
* As a baseline a Feed-Forward Neural Network (FFNN) was implemented, with grid search to tune number of neurons per layer and dropout rate.
*  The best accuracy was yielded by 128 neurons in the first layer, 64 neurons in the second
layer and a dropout rate of 0.1, which had accuracy of 0.64.

* A Convolutional Neural Network (CNN) was then implemented, using two 3×3 convolutions with 2×2 maximum pooling at each step. In the first convolution we use 32 filters, and in the second we use 64.
* This resulted in accuracy of 0.72, a significant improvement on the feed-forward neural
network, as a result of exploiting the spatial relationship between pixels.

* After adding augmentation, the accuracy was increased 0.96, with the four mis-classified images being between visually similar
pairs, where the certainty was relatively low in each case, as shown by the table below.

<img width="600" alt="Screenshot 2026-09-18 at 20 45 18" src="https://github.com/user-attachments/assets/ef13af9a-a52e-4b86-91ba-d1f0d2822e9d" />

<img width="600"  alt="Screenshot 2026-09-18 at 20 45 29" src="https://github.com/user-attachments/assets/39d4adb9-5322-4c94-aa8b-2858d38364c1" />

## Generation

* A Variational Autoencoder (VAE) was first implemented with 2 latent dimensions, producing images as shown below. These images
somewhat resemble the training images, however they are relatively blurry so that features aren’t strongly distinguishable.The images produced
by the two dimensional latent space produced predictions only in classes B, D, E and F, such that A and C were
not represented. It had a mean confidence of 0.77 across the data.

* A 10D latent space was then used, improved by sampling from random rows of the original distribution and adding a small amount of noise. This resulted in improved confidence of 0.80 and a more even spread of image assignment across classes.
* The Euclidean distance was used to ensure images were not repeats of the training data.
<img align=center width="600"  alt="Screenshot 2026-09-18 at 20 54 28" src="https://github.com/user-attachments/assets/b76f1eba-c679-48d1-8b8c-9dc85229696b" />
<img width="600"  alt="Screenshot 2026-09-18 at 20 48 59" src="https://github.com/user-attachments/assets/6f9496eb-3ddc-4dcb-987f-6b7c19abb1a7" />

* A GAN model was implemented however suffered mode collapse as the generator learned a pattern to fool the discriminator, though it bared little resemblance to the images.

