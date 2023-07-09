# Image-morphing-with-WGAN
Wasserstein Generative Adversarial Network (WGAN) used to create realistic images of human faces. It learns from a dataset of real face images and trains a Generator model to produce new, unique faces. 

This program implements a Wasserstein Generative Adversarial Network (WGAN) with gradient penalty for generating realistic human face images. It uses the CelebA dataset, which contains a large collection of labeled face images.

The program is written in Python and utilizes the PyTorch library for deep learning and image processing tasks. It also utilizes other libraries such as torchvision, PIL, matplotlib, and tqdm for data loading, image manipulation, visualization, and progress tracking.

The key components and steps of the program are as follows:

Importing Libraries: The necessary libraries and packages, including PyTorch, torchvision, and others, are imported at the beginning of the program.

Defining Helper Functions: The program defines a helper function show() to display images during training. This function shows a grid of images using matplotlib.

Setting Hyperparameters: The program sets various hyperparameters such as the number of epochs, batch size, learning rate, and the dimensionality of the latent space (z_dim). These values can be adjusted to customize the training process.

Setting Up Weights & Biases (wandb): The program optionally utilizes the Weights & Biases library for experiment tracking and visualization. It installs the library, logs in using a provided key, and initializes a new run with configuration parameters.

Defining the Generator and Critic Models: The program defines the Generator and Critic models using convolutional neural networks (CNNs) implemented in PyTorch. The Generator takes random noise as input and generates fake face images, while the Critic aims to distinguish between real and fake images.

Loading and Preparing the Dataset: The program downloads the CelebA dataset from a Google Drive link and extracts the image files. It defines a custom Dataset class to load and preprocess the images for training. The Dataset class converts the images to the desired size, normalizes pixel values, and prepares them for training.

Setting Up the DataLoader: The program sets up a DataLoader to efficiently load and iterate over batches of images during training. It uses the custom Dataset class and specifies the batch size.

Initializing Models and Optimizers: The program initializes the Generator and Critic models, as well as the corresponding Adam optimizers for updating their weights during training. The weights of the models are initialized using a custom init_weights function.

Defining the Gradient Penalty Function: The program defines a function to calculate the gradient penalty, which is used to enforce a smoothness constraint on the critic's outputs. This penalty term helps stabilize the training process of WGANs.

Saving and Loading Checkpoints: The program provides functions to save and load checkpoints of the trained models. These checkpoints include the model state dictionaries and optimizer states, allowing the models to be resumed or used for inference at a later time.

Training Loop: The program enters a training loop that iterates over the specified number of epochs. Within each epoch, it iterates over the batches of images from the DataLoader. The training loop alternates between updating the Critic and the Generator models.

Critic Training Step: The program performs the training step for the Critic model. It calculates the loss by comparing the predictions of the Critic on real and fake images, incorporating the gradient penalty. The loss is backpropagated through the Critic's network, and the optimizer updates the Critic's weights.

Generator Training Step: The program performs the training step for the Generator model. It generates fake images using random noise as input to the Generator, passes them through the Critic, and calculates the loss based on the Critic's predictions. The loss is backpropagated through the Generator's network, and the optimizer updates the Generator's weights.

Logging and Visualization: During training, the program optionally logs various metrics, such as the generator loss, critic loss, and epoch number, using the wandb library. It also visualizes the generated images and plots the generator and critic losses over time.

Saving Checkpoints and Visualizing Output: The program periodically saves checkpoints of the models at specified intervals and displays the generated images at regular intervals during training. It also plots the generator and critic losses over time to track their convergence.

Generating New Faces: After training, the program generates new face images using the trained Generator model. It generates synthetic faces by sampling random noise vectors and passing them through the Generator. The generated images can be displayed or saved for further analysis.

Face Morphing: The program demonstrates a face morphing capability by interpolating between two randomly sampled latent vectors. It creates a grid of intermediate face images that smoothly blend the features of the two faces.

The program provides a comprehensive pipeline for training a WGAN with gradient penalty on the CelebA dataset, generating new face images, and performing face morphing. It is designed to be flexible and extensible, allowing further customization and experimentation. You can use this program as a starting point for your own experiments or as a reference for understanding and implementing WGANs in PyTorch.
